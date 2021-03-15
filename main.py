import numpy as np
from knn import KNN

def main():
    DATASET_DIR = 'knn-sheets.csv'
    DATA_UJI_DIR = 'data-uji.csv'
    k_value = 3
    knn = KNN(DATASET_DIR,DATA_UJI_DIR,k_value)


if __name__ == '__main__':
    main()

