#!/usr/bin/env python3
"""
Sort Waymo E2E Driving TFRecords into per-segment files.

Example:
  python sort_waymo_e2e_by_uuid.py \
      --input_dir /path/to/e2e/train \
      --output_dir /path/to/e2e/train_sorted

What it does:
  - Reads all *.tfrecord* files in input_dir (recursively).
  - Parses each record as E2EDFrame.
  - Extracts a segment id (UUID) robustly:
      * If frame.context.name looks like "<uuid>-<index>" or "<uuid>_<index>",
        uses <uuid> as segment id and <index> as sort key.
      * Else uses frame.context.name as segment id and frame.timestamp_micros as sort key.
  - Writes one TFRecord per segment id in output_dir, with frames sorted by key.

Notes:
  - This script uses a two-pass "bucket then group" approach so it scales.
  - Supports compression_type '' or 'GZIP' (auto-detect default behavior is best-effort).

"""

import argparse
import os
import re
import json
import hashlib
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Optional, Tuple, Dict, List

import tensorflow as tf

# ----------------------------
# Proto import (robust)
# ----------------------------
def import_e2ed_proto():
    """
    Tries common proto module names used by Waymo Open Dataset installs.
    """
    candidates = [
        ("waymo_open_dataset.protos", "end_to_end_driving_data_pb2"),
        ("waymo_open_dataset.protos", "end_to_end_driving_pb2"),
        ("waymo_open_dataset.protos", "end_to_end_driving_dataset_pb2"),
    ]
    last_err = None
    for pkg, mod in candidates:
        try:
            module = __import__(f"{pkg}.{mod}", fromlist=[mod])
            # Must contain E2EDFrame
            getattr(module, "E2EDFrame")
            return module
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import an E2EDFrame proto. Tried: "
        + ", ".join([f"{a}.{b}" for a, b in candidates])
        + f"\nLast error: {last_err}"
    )

E2ED_PB2 = import_e2ed_proto()


# ----------------------------
# Helpers
# ----------------------------
UUID_IDX_RE = re.compile(r"^([0-9a-fA-F]{32})[-_](\d+)$")

def extract_segment_and_key(frame_name: str, timestamp_micros: int) -> Tuple[str, int]:
    """
    Returns:
      segment_id: str
      sort_key: int

    If frame_name matches "<32hex>-<index>" or "<32hex>_<index>":
      segment_id = uuid, sort_key = index
    else:
      segment_id = frame_name, sort_key = timestamp_micros
    """
    m = UUID_IDX_RE.match(frame_name)
    if m:
        seg = m.group(1).lower()
        idx = int(m.group(2))
        return seg, idx
    # fallback: use name as group id, and timestamp as ordering
    return frame_name, int(timestamp_micros)


def list_tfrecord_files(input_dir: str) -> List[str]:
    exts = (".tfrecord", ".tfrecords", ".tfrecord.gz", ".tfrecord.gzip")
    files = []
    for p in Path(input_dir).rglob("*"):
        if p.is_file() and p.name.endswith(exts):
            files.append(str(p))
    files.sort()
    return files


def try_open_dataset(files: List[str], compression_type: str):
    return tf.data.TFRecordDataset(files, compression_type=compression_type)


def detect_compression(files: List[str]) -> str:
    """
    Best-effort detection: try '' first, then 'GZIP'.
    """
    for c in ["", "GZIP"]:
        try:
            ds = try_open_dataset(files[:1], c)
            # force read one record
            for _ in ds.take(1):
                pass
            return c
        except Exception:
            continue
    # default
    return ""


def stable_bucket(segment_id: str, num_buckets: int) -> int:
    h = hashlib.md5(segment_id.encode("utf-8")).hexdigest()
    return int(h, 16) % num_buckets


class LRUWriterCache:
    """
    Keep only a limited number of TFRecordWriter objects open.
    """
    def __init__(self, max_open: int):
        self.max_open = max_open
        self._cache: "OrderedDict[str, tf.io.TFRecordWriter]" = OrderedDict()

    def get(self, path: str, compression_type: str = "") -> tf.io.TFRecordWriter:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        # evict if needed
        while len(self._cache) >= self.max_open:
            old_path, old_w = self._cache.popitem(last=False)
            old_w.close()

        opts = None
        if compression_type == "GZIP":
            opts = tf.io.TFRecordOptions(compression_type="GZIP")
        w = tf.io.TFRecordWriter(path, options=opts)
        self._cache[path] = w
        return w

    def close_all(self):
        for _, w in self._cache.items():
            w.close()
        self._cache.clear()


def make_tmp_example(segment_id: str, sort_key: int, proto_bytes: bytes) -> bytes:
    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                "segment_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[segment_id.encode("utf-8")])),
                "sort_key": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(sort_key)])),
                "proto": tf.train.Feature(bytes_list=tf.train.BytesList(value=[proto_bytes])),
            }
        )
    )
    return ex.SerializeToString()


def parse_tmp_example(serialized: bytes) -> Tuple[str, int, bytes]:
    ex = tf.train.Example()
    ex.ParseFromString(serialized)
    seg = ex.features.feature["segment_id"].bytes_list.value[0].decode("utf-8")
    key = int(ex.features.feature["sort_key"].int64_list.value[0])
    pb = ex.features.feature["proto"].bytes_list.value[0]
    return seg, key, pb


# ----------------------------
# Main pipeline
# ----------------------------
def bucketize(input_files: List[str],
              tmp_dir: str,
              num_buckets: int,
              input_compression: str,
              max_open_writers: int) -> List[str]:
    os.makedirs(tmp_dir, exist_ok=True)
    bucket_paths = [os.path.join(tmp_dir, f"bucket-{i:05d}.tfrecord") for i in range(num_buckets)]

    cache = LRUWriterCache(max_open=max_open_writers)

    ds = tf.data.TFRecordDataset(input_files, compression_type=input_compression)
    n = 0
    for rec in ds:
        pb = bytes(rec.numpy())
        frame = E2ED_PB2.E2EDFrame()
        frame.ParseFromString(pb)

        # Extract fields
        frame_name = getattr(frame.frame.context, "name", "")
        ts = int(getattr(frame.frame, "timestamp_micros", 0))

        seg, key = extract_segment_and_key(frame_name, ts)
        b = stable_bucket(seg, num_buckets)

        w = cache.get(bucket_paths[b], compression_type="")  # tmp buckets uncompressed
        w.write(make_tmp_example(seg, key, pb))

        n += 1
        if n % 20000 == 0:
            print(f"[bucketize] processed {n} records...")

    cache.close_all()
    print(f"[bucketize] done. total records: {n}")
    return bucket_paths


def write_sorted_segments(bucket_paths: List[str],
                          output_dir: str,
                          output_compression: str) -> Dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    counts: Dict[str, int] = {}
    total = 0

    for bi, bpath in enumerate(bucket_paths):
        if not os.path.exists(bpath) or os.path.getsize(bpath) == 0:
            continue

        groups: Dict[str, List[Tuple[int, bytes]]] = defaultdict(list)

        ds = tf.data.TFRecordDataset([bpath], compression_type="")
        for rec in ds:
            seg, key, pb = parse_tmp_example(bytes(rec.numpy()))
            groups[seg].append((key, pb))

        # Write each segment from this bucket
        for seg, items in groups.items():
            items.sort(key=lambda x: x[0])
            out_path = os.path.join(output_dir, f"{seg}.tfrecord" + (".gz" if output_compression == "GZIP" else ""))

            opts = None
            if output_compression == "GZIP":
                opts = tf.io.TFRecordOptions(compression_type="GZIP")

            # append if segment appears across multiple buckets (shouldn't, but safe)
            mode = "ab" if os.path.exists(out_path) else "wb"
            # TFRecordWriter doesn't support append mode directly; do manual write if appending
            if mode == "wb":
                w = tf.io.TFRecordWriter(out_path, options=opts)
                for _, pb in items:
                    w.write(pb)
                w.close()
            else:
                # Append: open a temp writer, then concatenate (still safe; slightly slower).
                # In practice, stable_bucket makes this extremely unlikely.
                tmp_append = out_path + ".append_tmp"
                w = tf.io.TFRecordWriter(tmp_append, options=opts)
                for _, pb in items:
                    w.write(pb)
                w.close()
                with open(out_path, "rb") as f1, open(tmp_append, "rb") as f2, open(out_path + ".merged", "wb") as fo:
                    fo.write(f1.read())
                    fo.write(f2.read())
                os.replace(out_path + ".merged", out_path)
                os.remove(tmp_append)

            counts[seg] = counts.get(seg, 0) + len(items)
            total += len(items)

        print(f"[write] bucket {bi+1}/{len(bucket_paths)}: wrote {len(groups)} segments")

    print(f"[write] done. total records written: {total} across {len(counts)} segments")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_buckets", type=int, default=512,
                        help="More buckets => less RAM per bucket pass, more files. 256-2048 reasonable.")
    parser.add_argument("--max_open_writers", type=int, default=64,
                        help="Max simultaneously open writers during bucketization.")
    parser.add_argument("--input_compression", type=str, default="AUTO", choices=["AUTO", "", "GZIP"])
    parser.add_argument("--output_compression", type=str, default="", choices=["", "GZIP"])
    parser.add_argument("--keep_tmp", action="store_true", help="Keep temporary bucket files.")
    args = parser.parse_args()

    files = list_tfrecord_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found under: {args.input_dir}")

    if args.input_compression == "AUTO":
        comp = detect_compression(files)
    else:
        comp = args.input_compression

    print(f"[info] found {len(files)} input tfrecord files")
    print(f"[info] input compression: {comp!r}")
    print(f"[info] output compression: {args.output_compression!r}")
    print(f"[info] num_buckets: {args.num_buckets}")

    tmp_dir = os.path.join(args.output_dir, "_tmp_buckets")
    bucket_paths = bucketize(
        input_files=files,
        tmp_dir=tmp_dir,
        num_buckets=args.num_buckets,
        input_compression=comp,
        max_open_writers=args.max_open_writers,
    )

    seg_out_dir = os.path.join(args.output_dir, "segments")
    counts = write_sorted_segments(
        bucket_paths=bucket_paths,
        output_dir=seg_out_dir,
        output_compression=args.output_compression,
    )

    manifest_path = os.path.join(args.output_dir, "manifest_segment_counts.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {"input_dir": args.input_dir, "segments_dir": seg_out_dir, "segment_counts": counts},
            f, indent=2, sort_keys=True
        )
    print(f"[done] manifest: {manifest_path}")

    if not args.keep_tmp:
        # cleanup
        for p in bucket_paths:
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(tmp_dir):
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass
        print("[cleanup] removed tmp buckets (use --keep_tmp to preserve)")

    print(f"[output] per-segment tfrecords are in: {seg_out_dir}")


if __name__ == "__main__":
    main()
