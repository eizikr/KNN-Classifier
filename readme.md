# Classifier for Handwritten Hebrew Letters.

## Description:
    - This program was created as part of course 'Image processing and computer vision' (Homework no.3).
    - The program is using 'K-Nearest-Neighbor' algorithm to classify images of letters from HHD_0 dataset which consists of handwritten letters in Hebrew.
    - Steps:
        1. Pre-processing.
            - Creaete new folder with 3 folders for 'testing', 'training' and 'validation'.
            - 'testing' and 'validation' contain 10% of random images from the dataset each, 'training' contains the rest 80%.
            - Every image shuld be 32x32 and with binarization.
        2. Find the best k (The k value that gives the best results).
        3. Train the model on the K we found.
        4. Test the model on the letters.
        5. Create reports (results and confusion matrix).

## HHD Dataset
- Info about the dataset: [Click-Here](https://www.researchgate.net/publication/343880780_The_HHD_Dataset)

## Environment:
    This program was created to be use in windows 11 OS.
    To use this program you requied:
    -   Installed python 3.10.4 (requied).
    -   Install openCV package -> pip install opencv-contrib-python.
    -   Install scikit-learn package -> pip install scikit-learn.

## How to Run Your Program:
    - run the program whith 1 input (dataset path).
        * example: "python knn_classifier.py hhd_dataset".

ENJOY!
