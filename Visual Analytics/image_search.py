# parser
import argparse

# path tools
import sys, os
sys.path.append(os.path.join(".."))

# image processing
import cv2

# tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input 

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# pandas
import pandas as pd

# data analysis
import numpy as np
from numpy.linalg import norm 
from sklearn.neighbors import NearestNeighbors


# traverse through folder
def traverse_files_in_folder(path):
    files_in_folder = []
    for root, dirs, files in os.walk(path):
        for file in files:
            files_in_folder.append((file, path + "/" + str(file)))
            
    return files_in_folder


# extraction function
def extract_features(img_path, model):
    """
    Extract features from image data using pretrained model (e.g. VGG16)
    """
    # Define input image shape - remember we need to reshape
    input_shape = (224, 224, 3)
    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    # convert to array
    img_array = img_to_array(img)
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocess image - see last week's notebook
    preprocessed_img = preprocess_input(expanded_img_array)
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img)
    # flatten
    flattened_features = features.flatten()
    # normalise features
    normalized_features = flattened_features / norm(features)
    
    return flattened_features


# image search function
def image_search(image, as_dir = False):
    
    # load VGG16 model
    model = VGG16(weights = 'imagenet',
              pooling = 'avg',
              include_top = False,
              input_shape = (224,224,3))
    
    
    # save the embedding into the list
    feature_list = []
    
    # set path
    directory = "../input"
    
    # traverse input_files through input folder
    input_files = traverse_files_in_folder(directory)
    # sort files
    input_files = sorted(input_files, key=lambda file: file[0])
    # use only files that does not have "checkpoint" in filename
    input_files = [file for file in input_files if "checkpoint" not in file[0]]
    
    # if going through directory
    if as_dir:
        # then directory should be image
        directory = image
    
    # otherwise
    else:
        # create target image index with the first file in input_files with enumerate function
        target_image_index = next((index for index, file in enumerate(input_files) if file[0] == image), -1)
        # if target image index can't find first file
        if target_image_index == -1:
            # print
            print("image file not found!")
    
    # for every image file in the directory
    for input_file in input_files:
        features = extract_features(input_file[1], model)
        feature_list.append(features)
    
    # find nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=10, 
                                 algorithm='brute',
                                 metric='cosine').fit(feature_list)
    
    # create dataframe with target image and three most similar images
    df = pd.DataFrame(columns = ('Target_image','First_similar','Second_similar','Third_similar'))
    
    # if not going through directory
    if not as_dir:
        # calculate nearest neighbor for target image
        distances, indices = neighbors.kneighbors([feature_list[target_image_index]])
        
        # convert to list
        idxs = []
        for i in range(1,4):
            # save indices and distances as tuple
            idxs.append((indices[0][i], distances[0][i]))
    
        # find target image
        plt.imshow(mpimg.imread(input_files[target_image_index][1]))
        # save target image as .png
        plt.savefig("../output/" 
                    + input_files[target_image_index][0] 
                    + "_target_image.png")

        # find 3 most similar to target image
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(mpimg.imread(input_files[idxs[0][0]][1]))
        axarr[0].set_title(str(idxs[0][1]))
        
        axarr[1].imshow(mpimg.imread(input_files[idxs[1][0]][1]))
        axarr[1].title.set_text(str(idxs[1][1]))
        
        axarr[2].imshow(mpimg.imread(input_files[idxs[2][0]][1]))
        axarr[2].title.set_text(str(idxs[2][1]))


        # Save an image which shows the three most similar images with the calculated distance score.
        f.savefig("../output/" 
                  + input_files[target_image_index][0] 
                  + "_similar_images.png")
    
        # create list with target image and the three most smilar images
        list_data = [image, 
                     input_files[idxs[0][0]][0], 
                     input_files[idxs[1][0]][0], 
                     input_files[idxs[2][0]][0]]

        # convert list to dataframe
        df = pd.DataFrame(columns = ('Target_image','First_similar','Second_similar','Third_similar'))
        # set labels of index 0
        df.loc[0] = list_data

        # save as .csv file
        csv = df.to_csv(os.path.join('../output/' 
                                     + input_files[target_image_index][0] 
                                     + '_similar_images.csv'), encoding = "utf-8")
    
    # otherwise
    else:
        # print the length of feature_list
        print(len(feature_list))
        # print the length of input_files
        print(len(input_files))
        
        # iterate over indexes in input_files
        for j in range(len(input_files)):
            distances, indices = neighbors.kneighbors([feature_list[j]])
            
            # convert to list
            idxs = []
            for i in range(1,4):
                # save indices and distances
                idxs.append((indices[0][i], distances[0][i]))
            
            # create list with target image and the three most smilar images
            list_data = [input_files[j][0], 
                         input_files[idxs[0][0]][0], 
                         input_files[idxs[1][0]][0], 
                         input_files[idxs[2][0]][0]]
            
            # set labels of indexes in input_files
            df.loc[j] = list_data
            
            # save as .csv.file
            csv = df.to_csv(os.path.join('../output/similar_images.csv'), encoding = "utf-8")
            
    return df


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


# main function
def main():
    args = parse_args()
    if args["filename"] is not None:
        image_search(args["filename"])
    if args["directory"] is not None:
        image_search(args["directory"], True)
            
        
# python program to execute
if __name__ == "__main__":
    main()
