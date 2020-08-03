import pandas as pd
import numpy as np 
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

def CountHashtags(data):
    """
    count words that start with # given a data string
    """
    assert isinstance(data, str)

    count = len([s for s in data.split() if s.startswith('#')])
    return count


def CountMentions(data):
    """
    count words that start with @ given a data string
    """
    assert isinstance(data, str)

    count = len([s for s in data.split() if s.startswith('@')])
    return count

def RemoveHashtags(data):
    """
    Removes hashtags from a given string

    Assuming re library is imported
    """
    assert isinstance(data, str)

    hashtag_regex = '#[A-Za-z0-9]+'

    data = ' '.join(re.sub(hashtag_regex, ' ', data).split())

    return data


def RemoveMentions(data):
    """
    Removes mentions from a given string

    Assuming re library is imported
    """
    assert isinstance(data, str)

    mention_regex = '@[A-Za-z0-9]+'

    data = ' '.join(re.sub(mention_regex, ' ', data).split())

    return data


def CountStopWords(data, stop_words):
    """
    Given a text and a list of stop words, return count of step words in text
    """
    assert isinstance(data, str)
    assert isinstance(stop_words, set)

    count = len([s for s in data.split() if s in stop_words])
    return count

def RemoveStopWords(data, stop_words):
    """
    Given a test and a list of stop words, remove stop words from text
    """
    assert isinstance(data, str)
    assert isinstance(stop_words, set)

    data = ' '.join([s for s in data.split() if s not in stop_words])
    return data

def CountWords(data):
    """
    count number of words in a given text
    """
    assert isinstance(data, str)

    count = len([s for s in data.split()])
    return count

def CountChars(data):
    """
    count number of characters in a given text
    """
    assert isinstance(data, str)

    return len(data)

def GetAvgWordLength(data):
    """
    Given a text, return average word length
    """
    assert isinstance(data, str)

    words = data.split()

    Len = 0
    for w in words:
        Len += len(w)

    Ans = int(Len / len(words))
    return Ans

def GetNumericDigitsCount(data):
    """
    Given a text, count number of numerical digits (not imbedded)
    """
    assert isinstance(data, str)

    count = len([s for s in data.split() if s.isdigit()])
    return count

def CountContractions(data, contractions):
    """
    Given a text and map of contractions, return count of contractions in text

    Make sure all characters are lowercase
    """
    assert isinstance(data, str)
    assert isinstance(contractions, dict)

    count = len([s for s in data.split() if s in contractions])
    return count

def ExpandContractions(data, contractions):
    """
    Given a text, and a map of contractions, replace contraction in the given text

    Make sure all character are lowercase
    """
    assert isinstance(data, str)
    assert isinstance(contractions, dict)

    for key in contractions:
        value = contractions[key]
        data = data.replace(key, value)

    return data

def CountEmails(data):
    """
    Given a text, count number of emails

    Assuming re library is imported
    """
    assert isinstance(data, str)

    email_regex = '([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'

    count = len(re.findall(email_regex, data))
    return count

def RemoveEmails(data):
    """
    Given a text, remove all emails from it

    Assuming re library is imported
    """
    assert isinstance(data, str)

    email_regex = email_regex = '([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'

    data = re.sub(email_regex, '', data)

    return data

def ExtractEmails(data):
    """
    Given a text, return all emails in this text

    Assuming re library is imported
    """
    assert isinstance(data, str)

    email_regex = '([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'

    emails = re.findall(email_regex, data)

    return emails

def CountUrl(data):
    """
    Given a text, count Urls it contains

    Assuming re library is imported
    """
    assert isinstance(data, str)

    url_regex = '(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'

    count = len(re.findall(url_regex, data))

    return count

def RemoveUrls(data):
    """
    Given a text, remove Urls it contains

    Assuming re library is imported
    """
    assert isinstance(data, str)

    url_regex = '(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'

    data = re.sub(url_regex, '', data)

    return data

def ExtractUrls(data):
    """
    Given a text, return all Urls it contains

    Assuming re library is imported
    """
    assert isinstance(data, str)

    url_regex = '(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'

    urls = re.findall(url_regex, data)

    return urls

def RemoveSpecialChars(data):
    """
    Given a text, remove special (non-alphanumeric) characters

    Assuming re library is imported
    """
    assert isinstance(data, str)

    special_regex = '[^A-Z a-z 0-9-,.!?]+'

    data = re.sub(special_regex, '', data)

    return data

def RemoveMultipleSpaces(data):
    """
    Given a text, remove multiple whistepaces
    """
    assert isinstance(data, str)

    data = ' '.join(data.split())

    return data

def RemoveHTMLTags(data):
    """
    Given a text remove any existing html tags

    Assuming BeautifulSoup is imported from bs4
    """
    assert isinstance(data, str)

    data = BeautifulSoup(data, 'lxml').getText()

    return data

def RemoveAccentedChars(data):
    """
    Given a text, remove accented chars
    """
    assert isinstance(data, str)

    data = unicodedata.normalize('NFKD', data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return data

def CorrectSpelling(data):
    """
    Given a text, correct spelling of words using textblob

    Assuming TextBlob is imported from textblob
    """
    assert isinstance(data, str)
    data = TextBlob(data).correct()
    return data


def RemovePunctuations(data):
    """
    Given a text, remove its punctuation

    Assuming re is imported
    """
    assert isinstance(data, str)

    punc_regex = '[\.\,\!\?\:\;\-\=]'

    data = re.sub(punc_regex, '', data)
    return data

def AverageTextLengthPandas(data):
    """
    Given a pandas series, consisting of text input, calculate the average length of each text
    measured in characters
    """
    assert isinstance(data, pd.core.series.Series)
    Len = data.size
    Sum = sum([len(text) for text in data])

    return (Sum / Len)


def AverageWordPerTextPandas(data):
    """
    Given a pandas series, consisting of text input, calculate the average word length of each text
    """
    assert isinstance(data, pd.core.series.Series)

    Len = data.size
    Sum = sum([len(text.split()) for text in data])

    return (Sum / Len)

def CountWordsPandas(data):
    """
    Given a pandas series, consisting of text input, count total number of words
    """
    assert isinstance(data, pd.core.series.Series)

    Ans = sum([CountWords(text) for text in data])

    return Ans

def CountContractionsPandas(data, contractions):
    """
    Given a pandas series, consisting of text input, and contractions dictionary, count number
    of contractions present in the text, assuming all text is lower case.
    """
    assert isinstance(data, pd.core.series.Series)
    assert isinstance(contractions, dict)

    Ans = sum([CountContractions(text, contractions) for text in data])

    return Ans

def CountStopWordsPandas(data, stop_words):
    """
    Given a pandas series, consiting of text input, and a stop words set, count the number of
    stop words present in the series
    """
    assert isinstance(data, pd.core.series.Series)
    assert isinstance(stop_words, set)

    Ans = sum([CountStopWords(text, stop_words) for text in data])

    return Ans

def AllTextPandas(data):
    """
    Given a pandas series, consisting of text input, return all the text concatenated as a single
    string
    """
    assert isinstance(data, pd.core.series.Series)
    
    Text = ' '.join([text for text in data])

    return Text

def WordNetPOS(word):
  """
  Given a word return its POS tagging according to nltk
  """
  assert isinstance(word, str)
  tag = nltk.pos_tag([word])[0][1][0].upper()
  tag_dict = {"J" : wordnet.ADJ,
              "N" : wordnet.NOUN,
              "V" : wordnet.VERB,
              "R" : wordnet.ADV}
  
  return tag_dict.get(tag, wordnet.NOUN)

def LemmatizeWithWordNet(text):
  """
  Given a text, lemmatize this text using nltk
  Make sure you run the following before calling this method
  nltk.download('wordnet')
  nltk.download('averaged_perceptron_tagger')
  """
  assert isinstance(text, str)

  tokens = text.split()

  Lemmatizer = WordNetLemmatizer()

  text_lemmatized = ' '.join([Lemmatizer.lemmatize(word, WordNetPOS(word)) for word in tokens])

  return text_lemmatized

def SplitWithRegex(data, regex):
    """
    Given a text and a regex, split the text according to the delimiter specified by the regex
    """
    assert isinstance(data, str)
    valid_regex = False
    
    try:
        re.compile(regex)
        valid_regex = True
    except re.error:
        valid_regex = False
    
    assert valid_regex == True
    
    split_text = re.split(regex, data)
    
    return split_text

def GensimWord2Vec(data):
    """
    Given a text data, create a Word2Vec model using gensim library
    """
    assert isinstance(data, str)

    sentences = SplitWithRegex(data, '[?,.!]')
    
    sentences = [text.split() for text in sentences]
        
    phrases = Phrases(sentences, min_count = 30, progress_per = 10000)    
    
    bigram = Phraser(phrases)
    
    sentences = bigram[sentences]
     
    word2vec_model = Word2Vec(min_count = 1,
                              window = 5,
                              size = 300,
                              min_alpha = 0.0007,
                             negative = 20,
                             workers = 2)
    
    word2vec_model.build_vocab(sentences, progress_per = 10000)
    
    word2vec_model.train(sentences, 
                         total_examples = word2vec_model.corpus_count,
                         epochs = 30,
                         report_delay = 1)
    
    word2vec_model.init_sims(replace = True)
    
    return word2vec_model
    
def LemmatizeWithSpacy(text):
  """
  Given a text, lemmatize it using spacy libary
  Make sure you have called 
  nlp = spacy.load('en_core_web_lg')
  """
  if not isinstance(text, str):
    return ""
  doc = nlp(text)

  text_lemmatized = ' '.join([token.lemma_ for token in doc])

  return text_lemmatized
    
contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"btw": "by the way",
"cuz": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"gonna": "going to",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"sth": "something",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'll": "we will",
" u ": " you ",
" ur ": " your ",
" n ": " and ",
"you're": "you are"}

