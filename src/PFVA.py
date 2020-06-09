from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.decomposition import PCA
from LR_Functions import *
import numpy as np
import time
import cv2
import os

lengthOfDataset = 0
textDataset = []


def readTextPerImage(pathFolder, verbose=True):
    textInformationFilePath = pathFolder+"database_information.txt"
    with open(textInformationFilePath, 'r') as reading:
        fR = reading.read().split('\n')

    lengthOfDataset = len(fR)
    textDataset = [["" for x in range(7)] for y in range(lengthOfDataset)]

    for instanceNumber, instance in enumerate(fR):
        allData = instance.split(",")
        for dataColumn, data in enumerate(allData):
            textDataset[instanceNumber][dataColumn] = data
    if verbose:
        print(textDataset[0])
        print(textDataset[-1])
        print(lengthOfDataset)


def convertImageToArray(imageDatasetPath, lengthOfDataset, textDataset):
    imagesDatasetAsMatrix256 = []
    yLabel = []

    for instanceOfInterest in range(lengthOfDataset, verbose=True):
        print("\r[KRANOK] - Processing image (%d/%d)..." %
              (instanceOfInterest+1, lengthOfDataset), end='')
        img1024 = cv2.imread(
            imageDatasetPath+textDataset[instanceOfInterest][0]+".pgm", cv2.IMREAD_GRAYSCALE)
        img256 = cv2.resize(img1024, (256, 256), interpolation=cv2.INTER_AREA)

        x256 = np.array(img256)/255
        x256 = np.reshape(x256, img256.shape[0]*img256.shape[1])

        if(textDataset[instanceOfInterest][2] == "NORM"):
            yLabel.append(1)
        else:
            yLabel.append(0)

        imagesDatasetAsMatrix256.append(x256)
    if verbose:
        print("\n")
        print(len(imagesDatasetAsMatrix256), len(imagesDatasetAsMatrix256[0]))

        print("\n")
        print(imagesDatasetAsMatrix256[0])
    return imagesDatasetAsMatrix256


def calculatePCA(imagesDatasetAsMatrix256, verbose=True):
    A = imagesDatasetAsMatrix256.copy()
    A = np.array(A)
    M = np.mean(A, axis=0)

    C = A-M
    V = np.cov(C.T)
    values, vectors = np.linalg.eig(V)

    P = vectors.T.dot(C.T)
    return P


def calculateLogisticRegression(train_set_x, test_set_y, iterations=20000, rate=0.005, verbose=True):
    train_set_x = train_set_x.reshape(
        train_set_x.shape[1], train_set_x.shape[0])
    test_set_x = test_set_x.reshape(test_set_x.shape[1], test_set_x.shape[0])
    train_set_y = train_set_y.reshape(1, train_set_y.shape[0])
    test_set_y = test_set_y.reshape(1, test_set_y.shape[0])

    d = model(train_set_x, train_set_y, test_set_x, test_set_y,
              num_iterations=iterations, learning_rate=rate, print_cost=verbose)
    return d


def functionCalculatePCA(imagesDatasetAsMatrix256):
    A = imagesDatasetAsMatrix256.copy()
    pcaDimensions = 200  # int(len(A[0])/2)
    pca = PCA(pcaDimensions)
    pca.fit(A)

    P = (pca.transform(A)).T

    return P


def functionCalculateLogisticRegression(train_set_x, test_set_y):
    lm = linear_model.LogisticRegression()
    lRModel = lm.fit(train_set_x, train_set_y)
    print("Logistic Regression Library:",
          lRModel.score(test_set_x, test_set_y))
