# data analysis
import os
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")

# sentiment analysis VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# visualisations
import matplotlib.pyplot as plt


# get data
def load_data():
    filename = os.path.join("../input/fake_or_real_news.csv")
    data = pd.read_csv(filename)
    
    return data


# remove puncations
def remove_punc(string):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in string:  
        if ele in punc:  
            string = string.replace(ele, "") 
    
    return string


# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
        

# sentiment analysis of fake news
def analysis_fake(data): 
    # create dataframe with news labelled 'FAKE'
    fake_news_df = data[data["label"]=="FAKE"]

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

    return fake_news_df, vader_df_fake


# find GPEs for fake news
def gpe_fake(fake_news_df, vader_df_fake):
    # create empty list
    fake_gpe = []

    # for every headline, pipe through fake news df using the column "title"
    for headline in tqdm(nlp.pipe(fake_news_df["title"])):
        # create a list consisting of all GPE values
        headline_gpes = [x.text for x in headline.ents if x.label_ == "GPE"]
        # remove puncations in GPEs
        headline_gpes = [remove_punc(i) for i in headline_gpes]

        # check if the list has any values (meaning no GPEs)
        if not headline_gpes:
            # if no value, append "UNKNOWN" so we can easily remove all rows with "UNKNOWN" later
            fake_gpe.append("UNKNOWN")
            
        # check if the list only has 1 value (meaning one GPE)
        elif len(headline_gpes) == 1:
            # if one value, append the value to the list
            fake_gpe.append(headline_gpes[0])
            
        # remaining cases will be those with more than 1 value
        else:
            # start the string with the first [index 0] value of the list
            s = headline_gpes[0]

            # loop through the remaining part of the list and add a semicolons inbetween -
            # ... (starting from index 1 since we already assigned index 0 to s)
            for i in range(1, len(headline_gpes)):
                s += '; ' + headline_gpes[i]
            # append the string to fake_gpe
            fake_gpe.append(s)
                
    # create list with text ID, title, sentiment scores and GPE
    list_fake = list(zip(fake_news_df["Unnamed: 0"], 
                         fake_news_df["title"], 
                         vader_df_fake["neg"], 
                         vader_df_fake["neu"],
                         vader_df_fake["pos"], 
                         fake_gpe))

    # Convert to dataframe and name columns
    data_fake = pd.DataFrame(list_fake, columns = ["Text ID", 
                                                   "Title", 
                                                   "Negative", 
                                                   "Neutral", 
                                                   "Positive", 
                                                   "Geopolitical Entities"]) 

    # remove all rows with "UNKOWN" in GPE column
    data_fake = data_fake[data_fake["Geopolitical Entities"].str.contains("UNKNOWN") == False]
    
    # display new dataframe with fake news and GPEs
    print(data_fake)

    # convert and save to csv
    data_fake.to_csv("../output/output_fake.csv", encoding = "utf-8")
    
    
    return data_fake

def single_fake(fake_news_df):
    # create empty list
    single_fake = []

    # for every headline in tqdm pipe through fake news df using the column "title" and go through 3164 of them
    for headline in tqdm(nlp.pipe(fake_news_df["title"], batch_size=3164)):
        # and for every entity in these headlines
        for entity in headline.ents:
            # if that entity is labeled "GPE"
            if entity.label_ == "GPE":
                single_fake.append(entity.text)
                single_fake = [remove_punc(i) for i in single_fake]

    # convert to dataframe
    single_fake_df = pd.DataFrame(single_fake, columns = ["GPE"]) 
    
    return single_fake_df


# find top 20 GPEs in fake news
def top20_fake(single_fake_df):
    # count the frequency of each GPE in fake news
    fake_gpe_count = single_fake_df.value_counts('GPE')

    # take the 20 most common GPEs
    fake_gpe_top20 = fake_gpe_count.nlargest(20)

    # convert to list
    fake_top20 = fake_gpe_top20.tolist()

    #zip value and key
    fake_top20 = list(zip(fake_gpe_top20.index, fake_top20))

    return fake_top20


# create bar chart for top 20 GPEs in fake news
def chart_fake(fake_top20):
    # define pairs of x and y
    labels, y = zip(*fake_top20)
    x = np.arange(len(labels))
    y_ticks = list(range(0,150,10))
    
    # plot into bar chart
    plt.xticks(x, labels, rotation=75)
    plt.yticks(y_ticks)
    
    # add axes labels
    plt.xlabel("Geopolitical entities")
    plt.ylabel("Frequency")
    addlabels(x,y)
    
    # add title
    plt.title("Top 20 geopolitical entities in fake news")
    
    # plot as bar chart
    plt.bar(x, y, color = "red", width = 0.8)
    
    # add label values
    #for i in range(len(x)):
        #plt.text(i, y[i], y[i], ha = 'center')
    
    # save image
    plt.savefig(os.path.join("../output/chart_fake.png"), bbox_inches = 'tight')
    

# sentiment analysis for real news 
def analysis_real(data):
    # create dataframe with news labelled 'REAL'
    real_news_df = data[data["label"]=="REAL"]

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
    
    return real_news_df, vader_df_real


# find GPEs for real news
def gpe_real(real_news_df, vader_df_real):
    # create empty list
    real_gpe = []

    # for every headline, pipe through real news df using the column "title"
    for headline in tqdm(nlp.pipe(real_news_df["title"])):
        # create a list consisting of all GPE values
        headline_gpes = [x.text for x in headline.ents if x.label_ == "GPE"]
        # remove puncations in GPEs
        headline_gpes = [remove_punc(i) for i in headline_gpes]

        # check if the list has any values (meaning no GPEs)
        if not headline_gpes:
            # if no value, append "UNKNOWN" so we can easily remove all rows with "UNKNOWN" later
            real_gpe.append("UNKNOWN")
            
        # check if the list only has 1 value (meaning one GPE)
        elif len(headline_gpes) == 1:
            # if one value, append the value to the list
            real_gpe.append(headline_gpes[0])
            
        # remaining cases will be those with more than 1 value
        else:
            # start the string with the first [index 0] value of the list
            s = headline_gpes[0]

            # loop through the remaining part of the list and add a semicolons inbetween -
            # ... (starting from index 1 since we already assigned index 0 to s)
            for i in range(1, len(headline_gpes)):
                s += '; ' + headline_gpes[i]
            # append the string (e.g. 'Tehran;USA') to real_gpe
            real_gpe.append(s)
                
    # make a list with text ID, title, sentiment scores and GPE
    list_real = list(zip(real_news_df["Unnamed: 0"], 
                         real_news_df["title"], 
                         vader_df_real["neg"], 
                         vader_df_real["neu"], 
                         vader_df_real["pos"], real_gpe))

    # convert to dataframe and name columns
    data_real = pd.DataFrame(list_real, columns = ["Text ID", 
                                                   "Title", 
                                                   "Negative", 
                                                   "Neutral", 
                                                   "Positive", 
                                                   "Geopolitical Entities"]) 
    
    # remove all rows with "UNKOWN" in GPE column
    data_real = data_real[data_real["Geopolitical Entities"].str.contains("UNKNOWN") == False]
    
    # display new dataframe with real news and GPEs
    print(data_real)

    # convert and save to csv
    data_real.to_csv("../output/output_real.csv", encoding = "utf-8")
    
    return data_real


def single_real(real_news_df):
    # create empty list
    single_real = []

    # for every headline in tqdm pipe through real news df using the column "title" and go through 3164 of them
    for headline in tqdm(nlp.pipe(real_news_df["title"], batch_size=3164)):
        # and for every entity in these headlines
        for entity in headline.ents:
            # if that entity is labeled "GPE"
            if entity.label_ == "GPE":
                single_real.append(entity.text)
                single_real = [remove_punc(i) for i in single_real]

    # convert to dataframe
    single_real_df = pd.DataFrame(single_real, columns = ["GPE"]) 
    
    return single_real_df


# find top 20 GPEs in real news
def top20_real(single_real_df):
    # count the frequency of each GPE in real news
    real_gpe_count = single_real_df.value_counts('GPE')

    # take the 20 most common GPE
    real_gpe_top20 = real_gpe_count.nlargest(20)

    # convert to list
    real_top20 = real_gpe_top20.tolist()

    # zip value and key
    real_top20 = list(zip(real_gpe_top20.index, real_top20))
    
    return real_top20


# create bar chart for top 20 GPEs in real news
def chart_real(real_top20):
    # define pairs of x and y
    labels, y = zip(*real_top20)
    x = np.arange(len(labels))
    y_ticks = list(range(0,120,10))
    
    # plot into bar chart
    plt.xticks(x, labels, rotation=75)
    plt.yticks(y_ticks)
    
    # add axes labels
    plt.xlabel("Geopolitical entities")
    plt.ylabel("Frequency")
    addlabels(x,y)
    
    # add title
    plt.title("Top 20 geopolitical entities in real news")
    
    # plot as bar chart
    plt.bar(x, y, color = "red", width = 0.8)
    
    # add label values
    #for i in range(len(x)):
       # plt.text(i, y[i], y[i], ha = 'center')
    
    # save model
    plt.savefig(os.path.join("../output/chart_real.png"), bbox_inches = 'tight')
    
    
# main function
def main():
    data = load_data()
    
    # fake news
    fake_news_df, vader_df_fake = analysis_fake(data)
    data_fake = gpe_fake(fake_news_df, vader_df_fake)
    single_fake_df = single_fake(fake_news_df)
    fake_top20 = top20_fake(single_fake_df)
    chart_fake(fake_top20)
    
    # real news 
    real_news_df, vader_df_real = analysis_real(data)
    data_real = gpe_real(real_news_df, vader_df_real)
    single_real_df = single_real(real_news_df)
    real_top20 = top20_real(single_real_df)
    chart_real(real_top20)
    
    
# python program to execute
if __name__ == "__main__":
    main()