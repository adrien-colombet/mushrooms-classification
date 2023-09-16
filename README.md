# mushrooms-classification

## Repo structure

```bash
.
├── LICENSE
├── README.md
├── dataset
│   ├── images
│   ├── order_classfication
|   |   ├── edible_mushrooms.csv
│   │   ├── train
│   |   ├── test
│   │   └── validation
│   └── comestible_classification
│       ├── observations_mushroom.csv       
|       ├── train
│       ├── test
│       └── validation
└── nb
    ├── iteration_0 
    |   ├── dataset_widget.ipynb
    |   ├── edible_mushrooms_dataset_preparation.ipynb
    |   ├── edible_mushrooms_order_classification.ipynb
    |   └── images_exploration.ipynb
    ├── iteration_1 
```

Dataset contains data-frames:
* dataframe observation_mushooms.csv containing initial data
* data frame edible_mushrooms.csv containing clean data set for edible mushrooms classification

Directories:
* images : contains unstructured images
* first level : classification type
* second level : to store data for training and data for testing
* second level : classes for the specific classification problem

## Commit rules

* clear cell outputs before committing Jupyter notebooks
* insert comments and headers
* reusable custom functions should be integrated into a library
* do not commit images to maintain a lightweight repository