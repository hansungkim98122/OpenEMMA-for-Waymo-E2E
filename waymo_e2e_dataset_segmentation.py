import os 
import json
import argparse

def main(args):
    dataset_mode = args.dataset
    path_dir = args.dataset_dir + dataset_mode + '/'

    # Get all entry names (files and directories)
    entries = os.listdir(path_dir)

    segments = {}
    for entry in entries:
        split = entry.split('-')
        segment_UIUD = split[0]
        segment_num = split[1].split('.pkl')[0]
        if segment_UIUD in segments:
            segments[segment_UIUD].append(segment_num)
        else:
            segments[segment_UIUD] = [segment_num]

    #sort by segment_num
    for key in segments.keys():
        segments[key].sort()

    #save segments dict as json
    filename = 'waymo_' + dataset_mode + "_segments.json"
    try:
        with open(filename, 'w') as json_file:
            json.dump(segments, json_file, indent=4) # 'indent' makes the file human-readable
        print(f"Dictionary successfully saved to {filename}")
    except TypeError as e:
        print(f"Error saving file: {e}. Check if all dictionary values are JSON serializable types (strings, numbers, booleans, lists, or None).")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="testing")
    parser.add_argument("--dataset-dir", type=str, required=True, description='Path to Waymo E2E Preprocessed Dataset')
    args = parser.parse_args()
    main(args)