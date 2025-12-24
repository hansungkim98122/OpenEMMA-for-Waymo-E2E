#!/usr/bin/env python3
import os
import re
import json
import base64
import argparse
import tqdm
from math import atan2
import time
import numpy as np
import torch
from PIL import Image
import pickle
import json
import matplotlib.pyplot as plt


try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration, 
        MllamaForConditionalGeneration,
        AutoTokenizer,
    )
except Exception:
    AutoProcessor = None
    Qwen2VLForConditionalGeneration = None
    Qwen2_5_VLForConditionalGeneration = None
    MllamaForConditionalGeneration = None
    AutoTokenizer = None

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

# llava imports
try:
    from llava.model.builder import load_pretrained_model
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.conversation import conv_templates
except Exception:
    load_pretrained_model = None

# Configs
OBS_LEN = 16
FUT_LEN = 20
DT = 0.25
MAX_NEW_TOKENS = 1024
IMAGE_BUFFER_SIZE = 10

# Image helpers
def chw_rgb_to_pil(img_chw: np.ndarray) -> Image.Image:
    """img_chw: (3,H,W) RGB -> PIL RGB"""
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    if img_hwc.dtype != np.uint8:
        mx = float(img_hwc.max()) if img_hwc.size else 1.0
        if mx <= 1.0:
            img_hwc = (np.clip(img_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_hwc = np.clip(img_hwc, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(img_hwc, mode="RGB")

def pil_to_data_url(pil_img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    """Encode PIL image to data URL for OpenAI vision."""
    import io

    buff = io.BytesIO()
    pil_img.save(buff, format=fmt, quality=quality)
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"
def motion_summary_from_past(obs_ego_xy: np.ndarray, dt: float) -> dict:
    """
    Returns v_est, psi (tangent heading in ego frame), kappa estimate, turn_dir.
    psi=0 means straight ahead; y is left.
    """
    xy = np.asarray(obs_ego_xy, float)
    d = xy[1:] - xy[:-1]
    keep = np.linalg.norm(d, axis=1) > 1e-4
    d = d[keep]
    if len(d) < 3:
        return dict(v_est=0.0, psi=0.0, kappa=0.0, turn_dir="straight")

    v_est = float(np.linalg.norm(d[-1]) / dt)
    psi = float(np.arctan2(d[-1, 1], d[-1, 0]))

    psi_seq = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    ds = np.linalg.norm(d, axis=1)
    m = min(5, len(d))
    dpsi = float(psi_seq[-1] - psi_seq[-m])
    s = float(np.sum(ds[-m:]) + 1e-6)
    kappa = float(dpsi / s)  # signed

    turn_dir = "left" if kappa > 0.01 else ("right" if kappa < -0.01 else "straight")
    return dict(v_est=v_est, psi=psi, kappa=kappa, turn_dir=turn_dir)

# Frame normalization (works whether pose is already ego@t0 or not)
def normalize_to_ego_t0(xy: np.ndarray) -> np.ndarray:
    """
    Ensures last point is [0,0] and heading aligns with +x (forward).
    If your pose is already ego@t0 with x forward,y left, this is near-identity.
    """
    xy = np.asarray(xy, float)
    if xy.shape[0] < 2:
        return xy.copy()

    # Recenter so last is origin
    xy0 = xy - xy[-1]

    # Align last velocity with +x
    v = xy[-1] - xy[-2]
    yaw = atan2(v[1], v[0])  # current heading in this coordinate system
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]], dtype=float)  # rotate by -yaw
    xy1 = xy0 @ R.T
    return xy1

# VLM inference wrappers
def getMessage(prompt, image=None, args=None, sys_message=None):
    """
    Builds a chat message for different VLM backends.

    For Qwen: supports a list of PIL Images (temporal sequence).
    For GPT: handled elsewhere.
    """
    # Normalize image(s) into a list
    img_list = []
    if image is None:
        img_list = []
    elif isinstance(image, list):
        img_list = image
    else:
        img_list = [image]

    if "llama" in args.model_path or "Llama" in args.model_path:
        # Your llama path currently supports only one image in vlm_inference anyway.
        # We'll keep a single placeholder.
        message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        return message

    elif "qwen" in args.model_path or "Qwen" in args.model_path:
        message = []
        if sys_message:
            message.append({"role": "system", "content": [{"type": "text", "text": sys_message}]})

        # Add MULTIPLE images in order (oldest -> newest)
        content = []
        for im in img_list:
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": prompt})

        message.append({"role": "user", "content": content})
        return message
    return []

def vlm_inference(text=None, images=None, sys_message=None, processor=None, model=None, tokenizer=None, args=None, client=None):
    # LLaMA
    if ("llama" in args.model_path or "Llama" in args.model_path) and model is not None:
        image = images
        if isinstance(images, str):
            image = Image.open(images).convert("RGB")
        elif isinstance(images, Image.Image):
            image = images.convert("RGB")

        message = getMessage(text, args=args)
        input_text = processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=2048)
        output_text = processor.decode(output[0])

        m = re.findall(r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", output_text, re.DOTALL)
        return m[0].strip() if m else output_text.strip()

    # Qwen
    if ("qwen" in args.model_path or "Qwen" in args.model_path) and model is not None:
        if process_vision_info is None:
            raise RuntimeError("qwen_vl_utils.process_vision_info not available; install qwen-vl-utils or fix PYTHONPATH.")
        message = getMessage(text, image=images, args=args, sys_message=sys_message)
        chat_text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
        )

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0]

    # LLaVA
    if "llava" in args.model_path and model is not None:
        if load_pretrained_model is None:
            raise RuntimeError("llava not available in this environment.")

        conv_mode = "mistral_instruct"
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in text:
            if model.config.mm_use_im_start_end:
                text = re.sub(IMAGE_PLACEHOLDER, image_token_se, text)
            else:
                text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, text)
        else:
            if model.config.mm_use_im_start_end:
                text = image_token_se + "\n" + text
            else:
                text = DEFAULT_IMAGE_TOKEN + "\n" + text

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        image = images
        if isinstance(images, str):
            image = Image.open(images).convert("RGB")
        elif isinstance(images, Image.Image):
            image = images.convert("RGB")

        image_tensor = process_images([image], processor, model.config)[0]
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=2048,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # GPT (OpenAI API) — expects a client + key
    if "gpt" in args.model_path:
        if client is None:
            raise RuntimeError("OpenAI client not initialized. Install openai and provide a valid API key.")

        # Ensure images is a list of PIL or data URLs
        if isinstance(images, Image.Image):
            img_list = [images]
        elif isinstance(images, list):
            img_list = images
        else:
            img_list = [images]

        content = []
        for im in img_list:
            if isinstance(im, Image.Image):
                content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(im), "detail": "low"}})
            elif isinstance(im, str) and im.startswith("data:image"):
                content.append({"type": "image_url", "image_url": {"url": im, "detail": "low"}})
            else:
                # best effort: try treating as path
                try:
                    content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(Image.open(im).convert("RGB")), "detail": "low"}})
                except Exception:
                    pass

        content.append({"type": "text", "text": text})

        messages = []
        if sys_message is not None:
            messages.append({"role": "system", "content": sys_message})
        messages.append({"role": "user", "content": content})

        # Use chat.completions (your style)
        resp = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=messages,
            max_tokens=400,
        )
        return resp.choices[0].message.content

    raise RuntimeError(f"Unsupported model-path: {args.model_path}")

def SceneDescription(obs_image, processor=None, model=None, tokenizer=None, args=None, client=None):
    prompt = (
        "You are an autonomous driving assistant. You have access to this front-view camera image at time t0. "
        "Describe the driving scene focusing on traffic lights, stop signs, movements of other cars/pedestrians, and lane markings."
    )
    return vlm_inference(text=prompt, images=obs_image, processor=processor, model=model, tokenizer=tokenizer, args=args, client=client)

def DescribeObjects(obs_image, processor=None, model=None, tokenizer=None, args=None, client=None):
    prompt = (
        "You are an autonomous driving assistant. From this front-view image at time t0, "
        "list two or three road users you should pay attention to, their approximate location in the image, "
        "what they are doing, and why they matter."
    )
    return vlm_inference(text=prompt, images=obs_image, processor=processor, model=model, tokenizer=tokenizer, args=args, client=client)

def DescribeOrUpdateIntent(obs_image, prev_intent=None, processor=None, model=None, tokenizer=None, args=None, client=None):
    prompt = f"You are an autonomous driving assistant. Your intent is: {prev_intent}. Based on the current image, state your intent now. If you see a stop sign, slow down."
    return vlm_inference(text=prompt, images=obs_image, processor=processor, model=model, tokenizer=tokenizer, args=args, client=client)

# Waypoint generation
def GenerateMotion(obs_image, obs_ego_xy, given_intent,
                   processor=None, model=None, tokenizer=None, args=None, client=None):

    """
    obs_image: can be a PIL Image (single) OR list[PIL Image] (sequence oldest->newest).
    """

    # Normalize obs_image(s) to list
    if isinstance(obs_image, list):
        img_seq = obs_image
    else:
        img_seq = [obs_image]

    scene_description = object_description = intent_description = None
    if args.method == "openemma":
        scene_prompt = (
            "You are an autonomous driving assistant. You are given a TIME-ORDERED sequence of front-view images "
            "(oldest -> newest). The LAST image is the current time t0. "
            "Describe the scene at t0, using earlier frames to infer motion (traffic lights, other vehicles, pedestrians, lane markings). Especially, the current lane's curvature."
        )
        obj_prompt = (
            "You are an autonomous driving assistant. You are given a TIME-ORDERED sequence of front-view images "
            "(oldest -> newest). The LAST image is time t0. "
            "List 2-3 important road users at t0 and describe their motion using the earlier frames."
        )
        intent_prompt = (
            f"You are an autonomous driving assistant. You are given a TIME-ORDERED sequence of front-view images "
            f"(oldest -> newest). The LAST image is time t0. "
            f"The previous intent label is: {given_intent}, note that the ego vehicle may be in the middle of a turn."
            f"Based on the sequence (especially traffic lights and lane geometry), describe the ego vehicle's current intent from: turn left, stay in current lane, turn right."
        )

        scene_description = vlm_inference(
            text=scene_prompt, images=img_seq,
            processor=processor, model=model, tokenizer=tokenizer, args=args, client=client
        )
        object_description = vlm_inference(
            text=obj_prompt, images=img_seq,
            processor=processor, model=model, tokenizer=tokenizer, args=args, client=client
        )
        intent_description = vlm_inference(
            text=intent_prompt, images=img_seq,
            processor=processor, model=model, tokenizer=tokenizer, args=args, client=client
        )

        print(f"Scene Description: {scene_description}")
        print(f"Object Description: {object_description}")
        print(f"Intent Description: {intent_description}")

    # Kinematic summary from past waypoints
    summ = motion_summary_from_past(obs_ego_xy, DT)
    v_est, psi, kappa, turn_dir = summ["v_est"], summ["psi"], summ["kappa"], summ["turn_dir"]

    dx_est = float(v_est * DT)

    # stringify past
    obs_xy_str = ", ".join([f"[{p[0]:.2f},{p[1]:.2f}]" for p in np.asarray(obs_ego_xy)])

    sys_message = (
        f"You are an autonomous driving assistant.\n"
        f"You are given a TIME-ORDERED sequence of front-view images (oldest -> newest). "
        f"The LAST image is the current time t0.\n"
        f"You also receive the last {OBS_LEN} ego-frame waypoints of the ego vehicle relative to t0.\n\n"
        f"Ego-frame definition:\n"
        f"- x is forward (meters), y is left (meters)\n"
        f"- waypoint at t0 is [0,0]\n"
        f"- past waypoints are behind the vehicle (typically x <= 0)\n\n"
        f"Scene description: {scene_description}"
        f"Task: Predict the next {FUT_LEN} ego-frame waypoints for t0+dt,...,t0+{FUT_LEN}*dt, where dt = {DT} seconds.\n"
        f"Return ONLY valid JSON.\n"
    )

    prompt = f"""
    Past ego-frame waypoints (oldest->newest), dt={DT:.2f}s:
    {obs_xy_str}

    From past waypoints:
    - estimated speed v≈{v_est:.2f} m/s so typical forward step dx≈{dx_est:.2f} m (unless stopping)

    TASK:
    Predict future motion that stays in the CURRENT LANE at t0 and follows the lane curvature seen in the LAST image.
    The lane may curve even if the past waypoints look straight.

    INTERNAL STEPS (do this silently):
    1) Look at the LAST image and identify the ego lane:
    - lane centerline direction ahead (does it bend left or right?)
    - approximate curvature strength (mild / medium / sharp)
    - use double-yellow line + curb/road edge to decide the ego lane region.
    
    2) Decide the lane-following curvature direction:
    - If the lane bends RIGHT ahead, cumulative y should become NEGATIVE over time (y is left).
    - If the lane bends LEFT ahead, cumulative y should become POSITIVE.
    - If the lane is straight, cumulative y stays near 0.
    3) Generate deltas that follow that curvature smoothly.

    RULES:
    - dx_i >= 0 and usually near {dx_est:.2f} if moving (smooth changes only).
    - dy_i must reflect the lane curvature from the LAST image (not from past waypoints).
    - Smoothness: dy_i should change gradually (no sudden jumps).
    - Lane-keeping: total lateral displacement |y| should usually stay within ~2.0 m unless clearly changing lanes. Leave equal margin to both boundaries
    - If the lane curves right, do NOT output all dy_i≈0.

    SELF-CHECK (do silently):
    After producing deltas, accumulate to points (x_k, y_k) and verify in the LAST image:
    1) The curve direction matches the lane centerline ahead.
    2) Points stay centered in the lane: not near the curb/right edge.
    3) Keep a visible gap to the curb/sidewalk; do not place points on the shoulder/parking area.
    If any point approaches the curb/edge, shift dy values slightly LEFT (increase y) while keeping curvature direction.
    - If the lane bends right, dy should be negative overall, BUT not so negative that you move into the curb; keep the lane-center offset.
    4) If following the current lane, estimate the left lane boundary and right lane boundary. Plan a path that stays approximately midway between them.

    OUTPUT ONLY JSON with EXACTLY {FUT_LEN} deltas:
    {{"deltas":[[dx1,dy1],[dx2,dy2],...,[dx{FUT_LEN},dy{FUT_LEN}]]}}
    """.strip()
    
    
    out = vlm_inference(
        text=prompt,
        images=img_seq,
        sys_message=sys_message,
        processor=processor, model=model, tokenizer=tokenizer, args=args,
        client=client
    )

    return out, scene_description, object_description, intent_description

def fix_deltas_to_len(deltas, fut_len, obs_ego_xy, dt):
    deltas = np.asarray(deltas, dtype=float)

    # sanitize shape
    if deltas.ndim != 2 or deltas.shape[1] != 2:
        deltas = deltas.reshape(-1, 2)

    n = deltas.shape[0]
    if n == fut_len:
        return deltas

    # estimate a reasonable forward dx for padding
    summ = motion_summary_from_past(obs_ego_xy, dt)
    dx_est = max(0.05, float(summ["v_est"] * dt))  # at least 5 cm step

    if n == 0:
        pad = np.tile(np.array([[dx_est, 0.0]], dtype=float), (fut_len, 1))
        return pad

    if n < fut_len:
        # pad with last delta but ensure forward progress
        last = deltas[-1:].copy()
        last[0, 0] = max(last[0, 0], dx_est)
        pad = np.repeat(last, fut_len - n, axis=0)
        deltas = np.vstack([deltas, pad])
    else:
        deltas = deltas[:fut_len]

    return deltas

def load_dict_from_pickle(path: str) -> dict:
    with open(path,'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="gpt")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--method", type=str, default="openemma")
    parser.add_argument("--dataset", type=str, default="testing")
    parser.add_argument("--dataset-dir", type=str, required=True)
    args = parser.parse_args()
    print(args.model_path)

    # Initialize OpenAI client (only if needed)
    client = None
    if "gpt" in args.model_path:
        if OpenAI is None:
            raise RuntimeError("openai package not found but --model-path is gpt.")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment for --model-path gpt.")
        client = OpenAI(api_key=api_key)

    model = processor = tokenizer = None

    # Load local VLMs
    try:
        if ("qwen" in args.model_path or "Qwen" in args.model_path) and AutoProcessor is not None:
            # Try Qwen2.5 first (local path)
            try:
                if Qwen2_5_VLForConditionalGeneration is None:
                    raise RuntimeError("Qwen2_5_VLForConditionalGeneration not available in transformers.")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "/root/OpenEMMA/models/Qwen2.5-VL-3B-Instruct",
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
                processor = AutoProcessor.from_pretrained("/root/OpenEMMA/models/Qwen2.5-VL-3B-Instruct")
                tokenizer = None
                print("Loaded local Qwen2.5-VL-3B-Instruct (flash attention).")
            except Exception as e:
                print("Falling back to Qwen2-VL-7B-Instruct:", e)
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    dtype=torch.bfloat16,
                    device_map="auto",
                )
                processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
                tokenizer = None
                print("Loaded Qwen2-VL-7B-Instruct.")

        elif "llava" in args.model_path and load_pretrained_model is not None:
            disable_torch_init()
            if args.model_path == "llava":
                tokenizer, model, processor, _ = load_pretrained_model(
                    "liuhaotian/llava-v1.6-mistral-7b", None, "llava-v1.6-mistral-7b"
                )
            else:
                tokenizer, model, processor, _ = load_pretrained_model(args.model_path, None, "llava-v1.6-mistral-7b")
            print("Loaded LLaVA.")

        elif ("llama" in args.model_path or "Llama" in args.model_path) and MllamaForConditionalGeneration is not None:
            # If you use LLaMA vision locally, set your path here.
            raise RuntimeError("LLaMA vision loading not configured in this script.")

    except Exception as e:
        print("Model load error:", e)

    #  Load the dataset (your pickle) 
    if args.dataset == 'testing':
        with open('waymo_testing_segments.json', 'r') as f:
            segments_json = json.load(f) # Reads and parses JSON from a file
    elif args.dataset == 'val':
        with open('waymo_val_segments.json', 'r') as f:
            segments_json = json.load(f) # Reads and parses JSON from a file
    else:
        NotImplementedError

    skip = True
    for id in tqdm.tqdm(segments_json.keys()):
        if id == '5d1ea6ef0e47b35dffa21e9d79220b72':
            skip = False
        if skip: continue
        outdir = os.path.join(f"{args.model_path}_results", args.method, args.dataset,id)
        print(outdir, ' created')
        os.makedirs(outdir, exist_ok=True)

        image_history = []
        for segment in segments_json[id]:
            data_dict = load_dict_from_pickle(args.dataset_dir + args.dataset + "/" + id + "-" + str(segment) + '.pkl')

            #  Build stitched image (front-left, front, front-right) 
            cam_list = [1, 0, 2]
            cam_front_data = []
            for cam in cam_list:
                cam_front_data.append(data_dict["image_frames"][cam, ...].numpy().squeeze(0))  # (3,H,W)
            cam_front_data = np.concatenate(cam_front_data, axis=-1)  # concat width
            curr_image = chw_rgb_to_pil(cam_front_data)
            image_history.append(curr_image)

        image_history = image_history[-IMAGE_BUFFER_SIZE:] #only use last  frames

        #  Pose / trajectory 
        #Use last frame only for ego pose/trajectory
        pose = data_dict["ego_history_xyz"]  # (T,3)
        xy = pose[:, :2].astype(float)

        # Normalize to ego@t0 so last observed is [0,0] and heading aligns +x
        xy_ego = xy.copy()  # already vehicle frame (x forward, y left)
        # Ensure t0 origin is exactly [0,0] at the last observed waypoint
        xy_ego = xy_ego - xy_ego[OBS_LEN-1:OBS_LEN, :]  # subtract t0 (broadcast)

        obs_ego_xy = xy_ego[:OBS_LEN, :]


        # Print the past waypoints you feed the model (debug)
        obs_xy_str = ", ".join([f"[{p[0]:.2f},{p[1]:.2f}]" for p in obs_ego_xy])
        print(f"Observed ego-frame waypoints (oldest->newest): {obs_xy_str}")

        prev_intent = data_dict.get("ego_intent", "UNKNOWN")

        #  Generate motion 
        st = time.time()
        prediction_text, scene_desc, obj_desc, intent_desc = GenerateMotion(
            obs_image=image_history,
            obs_ego_xy=obs_ego_xy,
            given_intent=prev_intent,
            processor=processor, model=model, tokenizer=tokenizer, args=args,
            client=client
        )
        print(f'Generating Motion took {time.time() - st} seconds')
        #  Parse JSON 
        m = re.search(r"\{.*\}", prediction_text, re.DOTALL)
        obj = json.loads(m.group(0))
        deltas = fix_deltas_to_len(obj.get("deltas", []), FUT_LEN, obs_ego_xy, DT)
        ego_wps = np.cumsum(deltas, axis=0)
        pred_xy = ego_wps[:FUT_LEN]

        print("model deltas len:", len(obj.get("deltas", [])), "-> after fix:", deltas.shape[0])

        #append z
        pred_xyz = np.zeros((FUT_LEN,3))
        pred_xyz[:,2] = pose[-1,2]
        pred_xyz[:,:2] = pred_xy
        print(f"Got {pred_xy.shape[0]} future waypoints. "
            f"pred y-range={np.ptp(pred_xy[:,1]) if pred_xy.size else 0:.2f}")

        #  Plot in ego frame (simple, interpretable) 
        if args.plot and plt is not None:
            plt.figure()
            plt.plot(obs_ego_xy[:, 0], obs_ego_xy[:, 1], "k.-", label="Past (ego@t0)")
            plt.plot(pred_xy[:, 0], pred_xy[:, 1], "b.-", label="Pred future (ego@t0)")
            plt.axis("equal")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(outdir, "ego_frame_pred_vs_gt.png"))
            plt.close()

        #  Save outputs 
        np.save(os.path.join(outdir, "pred_xyz.npy"), pred_xyz)

        print(pred_xy)
        with open(os.path.join(outdir, "logs.txt"), "w") as f:
            f.write(f"Scene Description: {scene_desc}\n")
            f.write(f"Object Description: {obj_desc}\n")
            f.write(f"Intent Description: {intent_desc}\n")
            f.write(f"Raw Prediction:\n{prediction_text}\n")

        print("Saved to:", outdir)


if __name__ == "__main__":
    main()
