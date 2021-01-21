Sentiment Analysis of Kenyan Media Text

Introduction

This project aims to compare sentiment around female and male leaders in the Kenyan media source, The Daily Nation. The sentiment analysis model can be used to predict the polarity or sentiment of sentences, scoring each input sentence from 0 (very negative) to 5 (very positive). 


Requirements

This predictor is built using the AllenNLP library and Real-World NLP. It is trained on the Stanford Sentiment Treebank.
Python 3.5+
Pytorch (1.5.0 or higher)
AllenNLP (1.0.0 or higher) 
pip install allennlp==1.0.0 allennlp-models==1.0.0


Scraping Sentences

The first file is used to scrape sentences from the Kenyan newspaper The Daily Nation. Two word lists were used, one with names of Kenya’s prominent female leaders, and another with names of prominent male leaders. Sentences containing elements of the word list, along with 1 sentence preceding and 1 sentence following, are scraped and added to the male or female input files. This is repeated for 3 year intervals from 1999 to 2018.


Predictor and Analysis

This predictor (LSTM-RNN model) is trained on the Stanford Sentiment Treebank. After training, test the predictor on sentences, such as “This is a happy movie!”. Predict sentiment scores for each male and female input files. Output scores are saved in CSV format. Once each interval and gender combination is complete, distribution of sentiment scores 0-5 are calculated for each file.


Further Resources

AllenNLP: https://demo.allennlp.org/sentiment-analysis
Real-World NLP Book: http://www.realworldnlpbook.com/
Real-World NLP: https://github.com/mhagiwara/realworldnlp

