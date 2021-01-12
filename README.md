# Automatic Text Summarization


#### -- Project Status: Active

## Project Intro/Objective

The purpose of this project is to shed light on the old challenge of text summarization and take a step in the right direction amid its recent popularity. A lot of the research nowadays focus on abstractive techniques that use deep learning models to create human-like summaries. The problem with this approach is that it can create social bias and made up facts. Part of data science is to not only to innovate but to be able to use the right tools, so here we propose focusing on more traditional extractive methods and exploring ways to create a summarizer that is reliable and suitable for enterprise. 

### Methods Used
* TextRank (Graph-based approach)
* Latent Semantic Analysis (Topic modelling approach)
* TF-IDF (Term Frequency Approach)
* Feature Engineering + Naive Bayes (Machine Learning Approach)

### Technologies
* Python
* Pandas
* Numpy
* Matplotlib
* Sci-kit Learn
* NLTK
* Networkx
* Streamlit
* Heroku

## Project Description

In this project, we use the newsroom dataset containing millions of articles and summaries written by authors of major publications. We create summaries using different methods for each article and compare their scores. Finally, we create a simple app that allows the user to input any article and generate an extractive summary. 

## Findings

The TextRank algorithm was able to acheive a ROUGE-2 (F1) score of over 50 which is extremely good according to scores documented in scientific literature. 



## ROUGE-2 Scores
| Extractive                       | ROUGE-2 (F1)   |
|----------------------------------|----------------|
| TextRank                         | 50.31    |
| Latent Semantic Indexing         | 33.33          |
| TF-IDF                           | 26.01          |
| Naive Bayes                      | 32.26          |



## Next Steps

* Explore reinforcement learning method proposed by OpenAI
* Explore neural-network based extractive summarization

