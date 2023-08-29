# mushrooms-classification

## Repo structure

```bash
.
├── LICENSE
├── README.md
├── dataset
│   ├── edible_mushrooms.csv
│   ├── family
│   │   ├── test
│   │   └── train
│   ├── images
│   ├── observations_mushroom.csv
│   ├── order
│   │   ├── test
│   │   └── train
│   └── species
│       ├── test
│       └── train
└── nb
    ├── dataset_widget.ipynb
    ├── edible_mushrooms_dataset_preparation.ipynb
    ├── edible_mushrooms_order_classification.ipynb
    └── images_exploration.ipynb
```

Dataset contains data-frames:
* dataframe observation_mushooms.csv containing initial data
* data frame edible_mushrooms.csv containing clean data set for edible mushrooms classification

Directories:
* images : contains unstructured images
* first row : classification type
* second row : classes for the specific classification problem
* third row : to store data for training and data for testing


## Commit rules

* clear cell outputs before committing Jupyter notebooks
* insert comments and headers
* reusable custom functions should be integrated into a library
* do not commit images to maintain a lightweight repository