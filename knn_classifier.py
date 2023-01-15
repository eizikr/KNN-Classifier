import os
import shutil
import sys
import cv2
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy
import csv

WHITE_COLOR = [225,225,225]
OUTPUT_DATASET_DIR_NAME = 'Processed Dataset'
INPUT_DATASET_DIR_NAME = sys.argv[1]

def preProcessing():
    '''Creates 'Processed Dataset' whith 'training','validation', and 'testing' folders that filled with precessed images from dataset'''
    def padding(image):
        '''Padding the image to be a square'''
        height, width, _ = image.shape
        if height == width: 
            return image
        padding_size = abs(height - width) // 2
        if height > width:
            return cv2.copyMakeBorder(image, 0, 0, padding_size, padding_size, cv2.BORDER_CONSTANT, value=WHITE_COLOR)
        return cv2.copyMakeBorder(image, padding_size, padding_size, 0, 0, cv2.BORDER_CONSTANT, value=WHITE_COLOR)

    def binarization(image):
        '''Make the image be only 0(black) and 255(white)'''
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(img_gray, (3,3), 0)
        _, binary_img = cv2.threshold(blur_img, 0, 225, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binary_img

    def resize32(image):
        '''Resize the image to be 32x32'''
        dimantion = [32,32]
        return cv2.resize(image, dimantion)

    def createProcessedDir(name):
        '''Create a folder with inside folders 0-26'''
        os.mkdir(OUTPUT_DATASET_DIR_NAME + '/' + name)
        for i in range(0,27):
            os.mkdir(OUTPUT_DATASET_DIR_NAME + '/' + name + '/' + str(i))

    def fillFolders():
        '''Fill all folders with processed images'''
        for dir in os.listdir(INPUT_DATASET_DIR_NAME):
            i=0
            folder = INPUT_DATASET_DIR_NAME +'/'+ dir
            list = os.listdir(folder)
            shuffle(list)
            for image_file in list:
                image = cv2.imread(folder + '/' + image_file)
                padded_image = padding(image)
                resized_image = resize32(padded_image)
                binari_image = binarization(resized_image)            
                if(i % 9 == 0):
                    cv2.imwrite(OUTPUT_DATASET_DIR_NAME + '/' + 'testing' + '/' + dir + '/' + image_file, binari_image)
                elif(i % 8 == 0):
                    cv2.imwrite(OUTPUT_DATASET_DIR_NAME + '/' + 'validation' + '/' + dir + '/' + image_file, binari_image)
                else:
                    cv2.imwrite(OUTPUT_DATASET_DIR_NAME + '/' + 'training' + '/' + dir + '/' + image_file, binari_image)
                i+=1
    
    # Remove old folder, create new one, and fill it with new images
    if OUTPUT_DATASET_DIR_NAME in os.listdir('./'):
        shutil.rmtree(OUTPUT_DATASET_DIR_NAME)
    os.mkdir(OUTPUT_DATASET_DIR_NAME)
    createProcessedDir('testing')
    createProcessedDir('validation')
    createProcessedDir('training')
    fillFolders()


def allVectorsAndLabels(folder):
    '''Returns vectors and labels arrays for all data in a folder'''
    vectors = []
    labels = []
    path = OUTPUT_DATASET_DIR_NAME + '/' + folder
    for dir in os.listdir(path):
        v,l = dirVectorsAndLables(folder, dir)
        vectors += v
        labels += l
    return vectors,labels

def dirVectorsAndLables(folder, dir):
    '''Returns vectors and labels arrays for specific directory'''
    vectors = []
    labels = []
    path = OUTPUT_DATASET_DIR_NAME + '/' + folder
    for image_file in os.listdir(path + '/' + dir):
        image = cv2.imread(path + '/' + dir + '/' + image_file)    
        vectors.append(image.flatten())
        labels.append(dir)
    return vectors,labels

def test_model(model,k):
    '''Test the model for every letter and save it to files 'results.txt' and 'Confusion Matrix.csv' '''
    f = open('results.txt','w')
    f.write(f'k = {k}\n')
    f.write(f'Letter\t\tAccuracy\n')
    true_labels = numpy.array([])
    predicts_labels = numpy.array([])

    for dir in range(0,27):
        vector, labels = dirVectorsAndLables('testing', f'{dir}')
        acc = model.score(vector, labels) * 100
        f.write(f'  {dir:<9}{acc:.2f}%\n')

        true_labels = numpy.concatenate((true_labels, labels))
        predicts_labels = numpy.concatenate((predicts_labels, model.predict(vector)))

    f.close()

    matrix = confusion_matrix(true_labels, predicts_labels, labels=[str(i) for i in range(0,27)])
    matrixToCSV('Confusion Matrix.csv', matrix)

def matrixToCSV(filename,matrix):
    '''Create csv file and fill it with matrix'''
    with open(filename,'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in matrix:
            writer.writerow(row)

def findBestK(train_vects, train_labels, valid_vects, valid_labels):
    '''Find the k value with the bess acc'''
    best_k = 1
    max_acc = 0
    for k in range(1,16,2):
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        model.fit(train_vects, train_labels)
        acc = model.score(valid_vects, valid_labels)
        if max_acc < acc:
            best_k = k
            max_acc = acc
    return best_k


def knn_classifier():
    # Pre Process all images    
    preProcessing()

    # Get Vectors and Labels for training and validation 
    train_vects,train_labels = allVectorsAndLabels('training')
    valid_vects,valid_labels = allVectorsAndLabels('validation')

    # Train the model with the best k value    
    max_k = findBestK(train_vects, train_labels, valid_vects, valid_labels)    
    model = KNeighborsClassifier(n_neighbors=max_k, metric='euclidean')
    model.fit(train_vects,train_labels)

    # Test the model (Create Result and Confusion Matrix)
    test_model(model,max_k)



if __name__ == "__main__":
    knn_classifier()