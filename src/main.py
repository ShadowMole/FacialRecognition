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


def magic(vectors, meanVector):
    for i in range(len(vectors)):
        vectors[i] = numpy.subtract(vectors[i], meanVector)
    npvectors = numpy.asarray(vectors)
    print(len(npvectors.transpose()))
    covMat = numpy.matmul(npvectors.transpose(), npvectors)
    covMat = numpy.divide(covMat, 5)
    #for i in range(5):
     #   print(covMat[i])
    eVals, eVects = numpy.linalg.eig(covMat)
    modifiedMergeSort(eVals, eVects)
    for i in range(20):
        print(eVals[i])

    p = numpy.matmul(eVects, npvectors.transpose())
    final = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(14):
        for j in range(19):
            final[i].append(p[j][i])
    return final, eVects


def smallmagic(vector, meanVector, eVects):
    vector = numpy.subtract(vector, meanVector)
    npvectors = numpy.asarray(vector)
    npvectors = npvectors.transpose()
    p = numpy.matmul(eVects, npvectors.transpose())
    final = []
    for j in range(19):
        final.append(p[j])
    return final


def train():
    meanVector = []
    vectors = []
    for i in range(5):
        path = 'C:\\Users\\steve\\PycharmProjects\\FacialRecognition\\src\\TrainingImages\\1_'
        imname = str(i+1)+'.jpg'
        # Hard Coded file path because it won't work otherwise
        # Path for Steven
        newpath = path + imname
        # Path for Jatin
        # path ='' + imname

        print(path)
        image = cv2.imread(newpath, 0)
        # cv2.imshow('hi', image)
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
    for i in range(5):
        path = 'C:\\Users\\steve\\PycharmProjects\\FacialRecognition\\src\\TrainingImages\\2_'
        imname = str(i+1)+'.jpg'
        # Hard Coded file path because it won't work otherwise
        # Path for Steven
        newpath = path + imname
        # Path for Jatin
        # path ='' + imname

        print(path)
        image = cv2.imread(newpath, 0)
        # cv2.imshow('hi', image)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vector = []
        n = 0;
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                vector.append(image[j, k] / 255)
                meanVector[n] = meanVector[n] + vector[n]
                n = n + 1
        vectors.append(vector)
    for i in range(5):
        path = 'C:\\Users\\steve\\PycharmProjects\\FacialRecognition\\src\\TrainingImages\\3_'
        imname = str(i+1)+'.jpg'
        # Hard Coded file path because it won't work otherwise
        # Path for Steven
        newpath = path + imname
        # Path for Jatin
        # path ='' + imname

        print(path)
        image = cv2.imread(newpath, 0)
        # cv2.imshow('hi', image)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vector = []
        n = 0;
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                vector.append(image[j, k] / 255)
                meanVector[n] = meanVector[n] + vector[n]
                n = n + 1
        vectors.append(vector)
    meanVector = numpy.divide(meanVector, 15)
    final, eVects = magic(vectors, meanVector)
    return meanVector,final, eVects


def test(image, meanVector, train, eVects, actual, num):
    vector = []
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            vector.append(image[j, k] / 255)
    vector = smallmagic(vector, meanVector, eVects)
    min = 10000000
    type = ''
    for i in range(14):
        diff = 0
        for j in range(19):
            x = vector[j] - train[i][j]
            diff += x * x
        diff = numpy.sqrt(diff)
        if diff < min:
            min = diff
            if i < 5:
                type = '1'
            elif i < 10:
                type = '2'
            else:
                type = '3'
    cv2.imwrite('C:\\Users\\steve\\PycharmProjects\\FacialRecognition\\src\\Tests\\' + str(num) + '_' + type + '_' + actual + '.jpg', image)

    #How do I show images in python with labels?
    #This is done except for getting all of the images in and showing output.


mean, train, eVects = train()
num = 1
for i in range(5):
    path = 'C:\\Users\\steve\\PycharmProjects\\FacialRecognition\\src\\TrainingImages\\'
    path = path + 'test1_' + str(i+1) + '.jpg'
    image = cv2.imread(path, 0)
    test(image, mean, train, eVects, '1', num)
    num += 1
for i in range(5):
    path = 'C:\\Users\\steve\\PycharmProjects\\FacialRecognition\\src\\TrainingImages\\'
    path = path + 'test2_' + str(i+1) + '.jpg'
    image = cv2.imread(path, 0)
    test(image, mean, train, eVects, '2', num)
    num += 1
for i in range(5):
    path = 'C:\\Users\\steve\\PycharmProjects\\FacialRecognition\\src\\TrainingImages\\'
    path = path + 'test3_' + str(i+1) + '.jpg'
    image = cv2.imread(path, 0)
    test(image, mean, train, eVects, '3', num)
    num += 1
