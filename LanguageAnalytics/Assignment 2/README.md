# Assignment 2 - Sentiment and NER
This folder contains a jypyter notebook for my second assignment as part of my portfolio exam in the course Language Analytics at Aarhus University taught by Ross Deans Kristensen-McLachlan in Spring 2022.  

The **goal** in this second assignment is to demonstrate that I have a good understanding of how to perform dictionary-based sentiment analysis and that I can use off-the-shelf NLP frameworks like spaCy to perform named entity recognition and extraction. 

I have chosen to do **task 2**, which contains a NER and sentiment analysis on the corpus of Fake vs Real news.  
More specific the tasks in task 2 is: 

 1. Split the data into two datasets - one of Fake news and one of Real news
 2. For every headline 
     * Get the sentiment scores
     * Find all mentions of geopolitical entites
     * Save a CSV which shows the text ID, the sentiment scores, and column showing all GPEs in that text
 3. Find the 20 most common geopolitical entities mentioned across each dataset - plot the results as a bar charts


The ```assignment2.ipynb``` is a small Python program that does the above. 

```output_fake.csv``` and ```output_real.csv``` in the ```output``` folder shows the results of the assignment. 

There are room for improvements and I have commented on these along the code.   
The improvements will be done before handing in my final submission in May 2022.
