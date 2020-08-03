import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

def CountHistogram(data):
    """
    Given a map of counts, visualize this map using a histogram
    """
    assert isinstance(data, dict)
    
    plt.bar(data.keys(), data.values())
    
def MostFrequentHistogram(data, rank):
    """
    Given a map of counts, and an integer rank, we will find the most 'rank' frequent words and visualize
    those as a histogram
    """
    assert isinstance(data, dict)
    assert isinstance(rank, int)
    
    counter = Counter(data)
    
    X = []
    Y = []
    
    for entry in counter.most_common(rank):
        X.append(entry[0])
        Y.append(entry[1])
    
    plt.bar(X,Y)
    
    return counter.most_common(rank)

def MakeWordCloud(data):
    """
    Given a string of words, create and plot a word cloud of all the words
    """
    assert isinstance(data, str)
    
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 1
    ).generate(data)
    
    plt.figure(figsize = (10, 10), facecolor = None) 
    plt.imshow(wordcloud, interpolation = 'bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 