
# import libraries
import streamlit as st

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", 
                        "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", 
                        "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                        "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", 
                        "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", 
                        "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", 
                        "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  
                        "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                        "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                        "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                        "mightn't": "might not","mightn't've": "might not have", "must've": "must have", 
                        "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 
                        "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", 
                        "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", 
                        "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 
                        "she'll": "she will", "she'll've": "she will have", "she's": "she is", 
                        "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", 
                        "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", 
                        "that'd've": "that would have", "that's": "that is", "there'd": "there would", 
                        "there'd've": "there would have", "there's": "there is", "here's": "here is",
                        "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                        "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                        "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 
                        "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 
                        "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 
                        "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", 
                        "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", 
                        "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                        "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
                        "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                        "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                        "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", 
                        "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                        "you'll've": "you will have", "you're": "you are", "you've": "you have"}

# create functions

def summarize(ranked_sentences, length):
    summary = ""
    for i in range(length):
        summary += ranked_sentences[i][1] + " "

    return summary

def rank(text, scores):
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(text)), reverse=True)
    ranked = ranked_sentences
        
    return ranked

def glove(cleaned_text):
    glove_vectors = []
    for i in cleaned_text:
        if len(i)!=0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
         
        glove_vectors.append(v)
        
    return glove_vectors

def clean(text):
    cleaned = []
    for y in text:
        cleaned.append(h_cleaner(y))
        
    return cleaned

def ready(text):
    cleaned = []
    for x in text:
        cleaned.append(s_cleaner(x))
        
    return cleaned

def s_cleaner(text): 
    final_string=[]
    for i in text:
        new_string = re.sub("\n", " ", text) 
        new_string = re.sub("Â\xad", "",new_string)
        new_string = re.sub(r'\([^)]*\)', '', new_string)
        new_string = re.sub(r'\[[^0-9]*\]', '', new_string)
        final_string.append(new_string)

    return new_string

def h_cleaner(text):   
    new_string = text.lower()
    new_string = re.sub(r'\([^)]*\)', '', new_string)
    new_string = re.sub('"','', new_string)
    new_string = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in new_string.split(" ")])    
    new_string = re.sub(r"'s\b","",new_string)
    new_string = re.sub("[^a-zA-Z]", " ", new_string) 
    new_string = [w for w in new_string.split() if not w in stop_words]
    new_string = " ".join(new_string)

    return new_string

# streamlit app

try:

    # title
    st.title('Text Summarizer')
    text = st.text_area("Input Text")

    # text preprocessing
    sentences = sent_tokenize(text)

    r = ready(sentences)

    c = clean(sentences)

    word_embeddings = {}
    with open('glove/glove.6B.100d.txt') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            word_embeddings[word] = coefs

<<<<<<< HEAD

=======
>>>>>>> 5a0f5da0d9a6626c1403fae639a0dbee595d81ae
    g = glove(c)

    # 1. intermediate representation
    sim_mat=np.zeros([len(g), len(g)])

    for y in range(len(g)):
        for x in range(len(g)):
            if y!=x:
                sim_mat[y][x] = cosine_similarity(g[y].reshape(1,100), g[x].reshape(1,100))[0,0]


    nx_graph = nx.from_numpy_array(sim_mat)
    pagerank_scores = nx.pagerank_numpy(nx_graph)

    # 2. rank sentences
    textranked = rank(r, pagerank_scores)

    # 3. sort by rank
    textrank_summary = summarize(textranked, 3)

    # display summary
    if text:
        st.write(textrank_summary)
        

except:
  # prevent the error from propagating into streamlit
  pass