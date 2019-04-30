import cv2
import numpy
import os

# Python program for implementation of MergeSort
def modifiedMergeSort(eVals, eVects):
    if len(eVals) > 1:
        mid = len(eVals) // 2  # Finding the mid of the eValsay
        valL = eVals[:mid]  # Dividing the eValsay elements
        valR = eVals[mid:]  # into 2 halves
        vectL = eVects[:mid]
        vectR = eVects[mid:]

        modifiedMergeSort(valL, vectL)  # Sorting the first half
        modifiedMergeSort(valR, vectR)  # Sorting the second half

        i = j = k = 0

        # Copy data to temp eValsays L[] and R[]
        while i < len(valL) and j < len(valR):
            if valL[i] < valR[j]:
                eVals[k] = valL[i]
                eVects[k] = vectL[i]
                i += 1
            else:
                eVals[k] = valR[j]
                eVects[k] = vectR[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(valL):
            eVals[k] = valL[i]
            eVects[k] = vectL[i]
            i += 1
            k += 1

        while j < len(valR):
            eVals[k] = valR[j]
            eVects[k] = vectR[j]
            j += 1
            k += 1

meanVector = []
vectors = []

for i in range(5):
    imname = '\\TrainingImages\\Steven'+str(i+1)+'.jpg'
    # Hard Coded file path because it won't work otherwise
    # Path for Steven
    path = 'C:\\Users\\steve\\PycharmProjects\\FacialRecognition\\src' + imname
    # Path for Jatin
    # path ='' + imname

    print(path)
    image = cv2.imread(path, 0)
    cv2.imshow('hi', image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vector = []
    n = 0;
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            vector.append(image[j, k] / 255)
            if i == 0:
                meanVector.append(vector[n])
            else:
                meanVector[n] = meanVector[n] + vector[n]
            n = n + 1
    vectors.append(vector)
meanVector = numpy.divide(meanVector, 5)
covMat = []
for i in range(len(vectors)):
    vectors[i] = numpy.subtract(vectors[i], meanVector)
    if i == 0:
        covMat = numpy.cov(numpy.stack((vectors[i], vectors[i]), axis=1))
    else:
        covMat = covMat + numpy.cov(numpy.stack((vectors[i], vectors[i]), axis=1))
covMat = numpy.divide(covMat, 5)
for i in range(30):
    print(covMat[i])
eVals, eVects = numpy.linalg.eig(covMat)
modifiedMergeSort(eVals, eVects)
for i in range(19):
     print(eVals[len(eVals) - 1 - i])
