# In terminal:
# cd cds-visual/portfolio/assignment_1/src
# python ImageSearch.py


# tips from lessons
    # lesson 11 - how to iterate over directory + Nearest neighbours
    # lesson 3 - cv2.calcHist function
        
    
# parser
import argparse

# path tools
import sys, os
sys.path.append(os.path.join(".."))

# image processing
import cv2

# utils
from utils.imutils import jimshow
from utils.imutils import jimshow_channel

# matplotlib
import matplotlib.pyplot as plt

# pandas
import pandas as pd

# data analysis
import numpy as np
from numpy.linalg import norm 
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


# get data for single file
def read_df(filename):
    filepath = os.path.join("..", 
                            "input",
                            filename)
    
    # load the data
    image = cv2.imread(filename)
    return data


# get data for file in folder
def read_file(filename):
    filepath = os.path.join(filename)
    
    # load the data
    image = cv2.imread(filename)
    return data


# traverse through folder
def traverse(path):
    files_in_folder = []
    for root, dirs, files in os.walk(path):
        for file in files:
            files_in_folder.append((file, path + "/" + str(file)))
    return files_in_folder


# parser
def parse_args():
    # Intialise argeparse
    ap = argparse.ArgumentParser()
    # command line parameters
    ap.add_argument("-fn", "--filename", required=False, help = "The filename to print")
    ap.add_argument("-d", "--directory", required=False, help = "The directory to print")
    # parse arguments
    args = vars(ap.parse_args())
    # return list of arguments
    return args


def compare_images():
# do this: Calculate the "distance" between the colour histogram of that image and all of the others.
# Calculate distance using the cv2.HISTCMP_CHISQR function in Open-CV
# Remember to normalize your images using something like MinMax
    
    # get histogram for target_image
    hist1 = cv2.calcHist([target_image],[0],None,[256],[0,256])
    # get histogram for comparison_image
    hist2 = cv2.calcHist([comparison_image],[0],None,[256],[0,256])
    # get score
    score=cv2.compareHist(hist1,hist2,cv2.HISTCMP_CHISQR)


def image_search(data, filename = None):
    if filename is None:
        output_image = "single_image.png"
        output_file = "single_image.csv"
    else:
        output_image = filename + "_similar.png"
        output_file = filename + "_similar.csv"

    model = VGG16(weights = 'imagenet',
              pooling = 'avg',
              include_top = False,
              input_shape = (224,224,3))
    
    
    # save the embedding into the list
    feature_list = []

    # for every image file in the directory
    for input_file in sorted(data):
        features = extract_features(input_file, model)
        feature_list.append(features)
    
    neighbors = NearestNeighbors(n_neighbors=10, 
                                 algorithm='brute',
                                 metric='cosine').fit(feature_list)
    
    distances, indices = neighbors.kneighbors([feature_list[250]])
    
    idxs = []
    for i in range(1,6):
        print(distances[0][i], indices[0][i])
        idxs.append(indices[0][i])
    
    # plot target image
    plt.imshow(mpimg.imread(data[250]))
    plt.savefig("../output/target_image.png")
    
    # plot 3 most similar
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(mpimg.imread(data[idxs[0]]))
    axarr[1].imshow(mpimg.imread(datajoined_paths[idxs[1]]))
    axarr[2].imshow(mpimg.imread(data[idxs[2]]))
    
    
# Save an image which shows the target image, the three most similar, and the calculated distance score.
    f.savefig("../output/Similar_images.png")
    

# Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order
    # create list 
    list_data = list(zip(target_image, similar1, similar2, similar3))
    
    # convert list to dataframe
    data_output = pd.DataFrame(list_data, columns = ['Target_image','First_similar','Second_similar','Third_similar'])

    # display new dataframe
    print(data_output)
    
    # save as csv
    csv = data_output.to_csv(os.path.join('../output/' + output_file), encoding = "utf-8")
    return data_output


# main function
def main():
    args = parse_args()
    if args["filename"] is not None:
        data = read_df(args["filename"])
        print(data)
        #analysis(data)
    if args["directory"] is not None:
        files_in_folder = traverse(args["directory"])
        print(files_in_folder)
        for file in files_in_folder:
            data = read_file(file[1])
            #analysis(data, file[0])
            
        
# python program to execute
if __name__ == "__main__":
    main()
