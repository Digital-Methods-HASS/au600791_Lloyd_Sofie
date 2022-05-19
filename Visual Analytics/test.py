# in terminal: 
# cd cds-lang/portfolio/assignment2/src
    # python SentimentNER.py -fn fake_or_real_news.csv

    
# Packages for data analysis
import os
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np

# Package for NLP
import spacy
nlp = spacy.load("en_core_web_sm")

# Packages for sentiment analysis VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Package for visualisations
import matplotlib.pyplot as plt

# parser
import argparse


def load_data():
    filename = os.path.join("../input/fake_or_real_news.csv")
    data = pd.read_csv(filename)
    
    return data


def analysis_fake(data): 
    # create dataframe with Fake news
    fake_news_df = data[data["label"]=="FAKE"]

    # let's have a look at our new pretty dataframe with all the fake news
    #return fake_news_df

    # create empty list
    vader_scores_fake = [] 

    # for every headline in the fake news df using the column named "title"
    for headline in fake_news_df["title"]: 
        # get the polarity score of the headline
        score = analyzer.polarity_scores(headline) 
        # append score to empty list
        vader_scores_fake.append(score) 
        
    # convert to dataframe
    vader_df_fake = pd.DataFrame(vader_scores_fake)
    
    # display
    # return vader_df_fake

    # create empty list
    fake_gpe = []

    # for every headline in tqdm, pipe through fake news df using the column "title" and go through 3164 of them
    for headline in tqdm(nlp.pipe(fake_news_df["title"], batch_size=3164)):
        # and for every entity in these headlines
        for entity in headline.ents:
            # if that entity is labeled "GPE"
            if entity.label_ == "GPE":
                fake_gpe.append(entity.text)
                
                
    # Make list with text ID, sentiment score and GPE
    list_fake = list(zip(fake_news_df["Unnamed: 0"], fake_news_df["title"], vader_df_fake["neg"], vader_df_fake["neu"], vader_df_fake["pos"], fake_gpe))

    # Convert to dataframe
    data_fake = pd.DataFrame(list_fake, columns = ["Text ID", "Title", "Negative", "Neutral", "Positive", "GPE"]) 

    # display new dataframe with fake news
    print(data_fake)

    # Convert to csv
    data_fake.to_csv("../output/output_fake.csv", encoding = "utf-8")
    
    # count the frequency of each GPE in fake news
    fake_gpe_count = data_fake.value_counts('GPE')

    # take the 20 most common GPE
    fake_gpe_top20 = fake_gpe_count.nlargest(20)

    # display
    #return fake_gpe_top20

    # convert to list
    fake_top20 = fake_gpe_top20.tolist()

    #zip value and key
    fake_top20 = list(zip(fake_gpe_top20.index, fake_top20))

    # display
    #return fake_top20

    # define pairs of x and y
    labels, y = zip(*fake_top20)
    x = np.arange(len(labels))
    y_ticks = list(range(0,100,10))
    # plot into bar chart
    plt.xticks(x, labels, rotation=75)
    plt.yticks(y_ticks)
    # add axes labels
    plt.xlabel("Geopolitical entities")
    plt.ylabel("Frequency")
    # add title
    plt.title("Top 20 geopolitical entities in fake news")
    # plot as bar chart
    plt.bar(x, y, color = "red", width = 0.8)
    # save image
    plt.savefig(os.path.join("../output/chart_real.png"))

    
def analysis_real(data):
    # create dataset with Real news
    real_news_df = data[data["label"]=="REAL"]

    # let's have a look at our real news!
    #return real_news_df

    # create empty list
    vader_scores_real = []

    # for every headline in the real news df using the column "title"
    for headline in real_news_df["title"]:
        # get the polarity score of the headline
        score = analyzer.polarity_scores(headline)
        # append score to empty list
        vader_scores_real.append(score)
        
    # convert to dataframe
    vader_df_real = pd.DataFrame(vader_scores_real)

    # Create empty list
    real_gpe = []

    # for every headline in tqdm, pipe through real news df using the column "title" and go through 3171 of them
    for headline in tqdm(nlp.pipe(real_news_df["title"], batch_size=3171)):
        # and for every entity in these headlines
        for entity in headline.ents:
            # if that entity is labeled "GPE"
            if entity.label_ == "GPE":
                # append to real_gpe
                real_gpe.append(entity.text)
                
    # Make a list with text ID, sentiment score and GPE
    list_real = list(zip(real_news_df["Unnamed: 0"], real_news_df["title"], vader_df_real["neg"], vader_df_real["neu"], vader_df_real["pos"], real_gpe))

    # Convert to dataframe
    data_real = pd.DataFrame(list_real, columns = ["Text ID", "Title", "Negative", "Neutral", "Positive", "GPE"]) 

    # display new dataframe with real news
    print(data_real)

    # convert to csv
    data_real.to_csv("../output/output_real.csv", encoding = "utf-8")
    
    # count the frequency of each GPE in real news
    real_gpe_count = data_real.value_counts('GPE')

    # take the 20 most common GPE
    real_gpe_top20 = real_gpe_count.nlargest(20)

    # convert to list
    real_top20 = real_gpe_top20.tolist()

    # zip value and key
    real_top20 = list(zip(real_gpe_top20.index, real_top20))

    # Define pairs of x and y
    labels, y = zip(*real_top20)
    x = np.arange(len(labels))
    y_ticks = list(range(0,100,10))
    # Plot into bar chart
    plt.xticks(x, labels, rotation=75)
    plt.yticks(y_ticks)
    # add axes labels
    plt.xlabel("Geopolitical entities")
    plt.ylabel("Frequency")
    # add title
    plt.title("Top 20 geopolitical entities in real news")
    # I would like to change the y axis using the following code, but I can't get it to work
    #plt.ylim(0,110)
    # plot as bar chart
    plt.bar(x, y, color = "red", width = 0.8)
    # save model
    plt.savefig(os.path.join("../output/chart_real.png"))
    
    
# main function
def main():
    data = load_data()
    analysis_real(data)
    analysis_fake(data) 
        
# python program to execute
if __name__ == "__main__":
    main()