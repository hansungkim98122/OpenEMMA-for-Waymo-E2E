# Waymo Open Motion Dataset 2025 Vision-based End-to-End Challenge
This repository is adopted from [OpenEMMA](https://github.com/taco-group/OpenEMMA) with my modifications to adapt to Waymo E2E dataset and challenge instead of NuScenes Dataset in its original implementation.


## Environment Setup:
The environment.yaml file that I used on my workstation is also included. I used CUDA 12.8 to support my GPU with Python 3.10. Try creating the conda environment
```
conda create -f environment.yaml
conda activate openemma_cu128
```
Alternatively, you can
```
pip install -r requirements.txt
```

See [OpenEMMA](https://github.com/taco-group/OpenEMMA) for detailed instructions on environment setup and dependency setup.

## Data Downloading:
You can download data from Google Cloud Buckets from [Waymo](https://waymo.com/open/download) website. The easiest way to download the data is to install gsutil and use the command
```
gsutil cp -r <bucket address. starts with gs:// > <destination folder directory>
```
## Data preprocessing:
After downloading the Waymo Open Motion Dataset (Waymo E2E). You can start preprocessing data, which generates pickle files for each scenario in the raw data proto. Note that training and test dataset pkl files are very large in size. So, you can choose the threshold number to limit the preprocessing.
```
waymo_data_preprocess.ipynb
```
The data directory must follow this hierarchy
```
├── <preprocessed_data_folder_name>
│   ├── val
│   │   ├── ...
│   ├── testing
│   │   ├── ...

```

## Data Sorting:
Since the raw data is not sorted by segment sequences and is randomly shuffled, we need to know the list of indices per segments to use the sequential WaymoE2EFrame() data. 
See [Waymo GitHub Issue](https://github.com/waymo-research/waymo-open-dataset/issues/921) for more details on this issue.
After preprocessing, run the following script to generate the json file that maps segment id to the list of indices.
```
python waymo_e2e_dataset_segmentation.py --data-dir <path-to-raw-data-folder> --dataset <dataset type: ['val','testing']>
```

## Inference Results Generation (Single Segment at the last frame): 
VLM models that are supported out-of-the-box are qwen and llava. You can use your OPENAI_API_KEY to use GPT-4 for inference. See [OpenEMMA](https://github.com/taco-group/OpenEMMA) for more details.
```
python main.py --model-path <qwen,gpt,llava> --dataset <testing or val> --dataset-dir <path to Waymo E2E preprocessed data folder>
```

# Inference for Visualization (All frames in a single segment):
```
python main_autoregressive.py --model-path <qwen,gpt,llava> --dataset <testing or val> --dataset-dir <path to Waymo E2E preprocessed data folder> --id <Segment id that you wish to run. Must exist in the dataset>
```

# Video Generation:
```
python generate_video.py --input <path to folder that contains all the results for the segment id per frame>
```

## Submission Generation:
See 
```
waymo_submission.ipynb
```

# Acknowledgement
The majority of the code is from [OpenEMMA](https://github.com/taco-group/OpenEMMA)

# License
All code in this repository is licensed under the Apache License 2.0.
