# mushrooms-classification

## Repo structure

```bash
.
├── app
├── dataset
│   ├── comestible_classification
│   │   ├── test
│   │   ├── train
│   │   └── validation
│   ├── mixed
│   │   ├── test
│   │   └── train
│   ├── new_order
│   │   ├── test
│   │   └── train
│   ├── order_classification
│   │   ├── test
│   │   ├── train
│   │   └── validation
│   └── wildfooduk
│       └── data
├── nb
│   ├── iteration_0
│   │   ├── logs
│   │   ├── model_densenet_checkpoint
│   │   ├── model_mobile_checkpoint
│   │   └── model_resnet_checkpoint
│   ├── iteration_1
│   │   ├── data
│   │   ├── model_mobile_segmented_checkpoint
│   │   ├── __pycache__
│   │   └── retinanet
│   └── iteration_2
│       ├── logs
│       └── model_mobile_mixed_checkpoint
└── raw
    ├── 0
    ├── 1
    ├── 2
    ├── 3
    ├── 4
    ├── 5
    ├── 6
    ├── 7
    └── images
```

Dataset contains data-frames:
* dataframe observation_mushooms.csv containing initial data
* data frame edible_mushrooms.csv containing clean data set for edible mushrooms classification

Directories:
* raw : contains unstructured images further split by id range
* dataset: dataset related to each classification type
* nb : containing notebooks subdivided into iterations to match project progression
* app : containing streamlit application 

## Commit rules

* clear cell outputs before committing Jupyter notebooks
* insert comments and headers
* reusable custom functions should be integrated into a library
* do not commit images to maintain a lightweight repository

## Problem breakdown

* first stage : mushrooms are classified to their edibility
* second stage : background removal with segment anything
* third stage : classification of edible mushrooms to their order
* fourth stage : mixed classification to get more information on the mushroom : family and genus whenever possible 

## Building the datasets

* raw images can be donwloaded from google drive via download_raw_images.ipynb
* binary problem dataset is built from binary_creation_datasets_entrainements.ipynb and binary_final_datasets_creation.ipynb
* edible problem dataset is built from edible_mushrooms_dataset_preparation.ipynb
* validation dataset for edible classification can be built from wild food uk import_wildfooduk_dataset.ipynb
