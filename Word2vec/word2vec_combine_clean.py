import subprocess
import os.path
import csv
import re
import random
import requests
from urllib.parse import urljoin
from urllib.request import urlopen as uReq
import nltk
import gensim
import pandas as pd;
import numpy as np;
import os;
import sys;
import multiprocessing;
from nltk.tokenize import word_tokenize
import os.path
from os import path
from svd2vec import svd2vec, FilesIO

startyr = 2019
endyr = 2019
jump = 1
subject = ["oped","lifestyle","news", "business", "counties", "sports"]
title = "oped_lifestyle_news_business_counties_sports"
#out_filename = "daily_nation_" + str(title) + "_" + str(startyr) + "_" + str(endyr) + "_combined.txt"
out_filename = "daily_nation_" + str(title) + "_" + str(startyr) + "_combined.txt"
print ("Creating combined text file...")
with open(out_filename, 'w') as f:
    for heading in subject:
        for yr in range(startyr, endyr+1, jump):
            newfile = 'daily_nation_' + str(heading) + "_" + str(yr)+ '_uncleaned.txt'
            if path.exists(newfile):
                with open(newfile, 'r', encoding="utf8") as fread:
                    print (str(yr))
                    for line in fread:
                        f.write(line)
f.close()

# Get text out of textfile
file = open(out_filename, "r") # Choose textfile to clean
articles = file.read()


###Preprocessing
### Cleaing the text
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

print (len(articles))
    
#Remove all characters except A-Z and a dot.
#alphas_only = re.sub("[^a-zA-Z\.\-]", " ", articles);
alphas_only = re.sub("[^a-zA-Z\.]", " ", articles);
        
#Normalize spaces to 1
multi_spaces = re.sub(" +", " ", alphas_only);
        
#Strip trailing and leading spaces
no_spaces = multi_spaces.strip();
        
#Normalize all charachters to lowercase
clean_text = no_spaces.lower();
        
#Get sentences from the tokenizer, remove the dot in each.
sentences = tokenizer.tokenize(clean_text);
sentences = [re.sub("[\.]", "", sentence) for sentence in sentences];
print(sentences[:50])
words = [nltk.word_tokenize(sent) for sent in sentences]
print(words[:50])


# Removing Stop Words and words from html
#print ("Removing stop words...")
#from nltk.corpus import stopwords
#stopwords = stopwords.words('english')
#print(stopwords)
#stopwords.extend(["p","mp","msonormal","strong","class","noprint","font","size","sh","img","alt","q","image","view","pic","jpg","th","em","br","k","w","g","v","also","fc","r","e","f","h","el","es","b","nstronger", "n"])
#stopwords.remove('she')
#stopwords.remove('hers')
#stopwords.remove('her')
#stopwords.remove('herself')
#stopwords.remove('he')
#stopwords.remove('his')
#stopwords.remove('him')
#stopwords.remove('himself')
                 
#for i in range(len(words)):
#    words[i] = [w for w in words[i] if w not in stopwords]
#print(words[:50])

## Creating a cleaned output file
print ("Creating cleaned text file...")
#out_cleaned_filename = "daily_nation_" + str(title) + "_" + str(startyr) +"_" + str(endyr) + "_cleaned_fin.txt" # Name output file
out_cleaned_filename = "daily_nation_" + str(title) + "_" + str(startyr) +"_" + "_cleaned_fin.txt" # Name output file
f = open(out_cleaned_filename, "w")
f.write(str(words))
f.close()

### First try at word2vec
word2vec = gensim.models.Word2Vec(words, min_count=5, window = 15, size = 150)
#svd2vec = svd2vec(words, size=150, window=15, min_count=5, verbose=False)
#save_as_svd = "daily_nation_" + str(title) + "_" + str(startyr) + "_" + str(endyr) + "_vectors_svd.txt"
#save_as_1 = "daily_nation_" + str(title) + "_" + str(startyr) + "_" + str(endyr) + "_vectors_check.txt"
#save_as_2 = "daily_nation_" + str(title) + "_" + str(startyr) + "_" + str(endyr) + "_vectors_check.bin"
#vocab = "daily_nation_" + str(title) + "_" + str(startyr) + "_" + str(endyr) + "_vocab_check.txt"
save_as_1 = "daily_nation_" + str(title) + "_" + str(startyr) + "_vectors_check.txt"
save_as_2 = "daily_nation_" + str(title) + "_" + str(startyr) + "_" + "_vectors_check.bin"
vocab = "daily_nation_" + str(title) + "_" + str(startyr) + "_" + "_vocab_check.txt"
#svd2vec.save_word2vec_format(save_as_svd)
word2vec.wv.save_word2vec_format(save_as_1, binary = False, fvocab = vocab)
word2vec.wv.save_word2vec_format(save_as_2, binary = True)
#word2vec.save("word2vec_1")
vocabulary = word2vec.wv.vocab
#print("Vectors: ")
#print(vocabulary)
sim_words = word2vec.wv.most_similar('woman', topn=25)
sim_words_2 = word2vec.wv.most_similar('man', topn=25)
sim_words_3 = word2vec.wv.most_similar('girl', topn=25)
sim_words_4 = word2vec.wv.most_similar('boy', topn=25)
sim_words_5 = word2vec.wv.most_similar('female', topn=25)
sim_words_6 = word2vec.wv.most_similar('male', topn=25)
#sim_words_7 = word2vec.wv.most_similar('depression', topn=25)
#sim_words_8 = word2vec.wv.most_similar('anxiety', topn=25)
sim_words_9 = word2vec.wv.most_similar('violence', topn=25)
sim_words_10 = word2vec.wv.most_similar('suicide', topn=25)
sim_words_11 = word2vec.wv.most_similar('leader', topn=25)
sim_words_12 = word2vec.wv.most_similar('addict', topn=25)
sim_words_13 = word2vec.wv.most_similar('alcoholic', topn=25)
sim_words_14 = word2vec.wv.most_similar('crash', topn=25)
sim_words_15 = word2vec.wv.most_similar('degree', topn=25)
sim_words_16 = word2vec.wv.most_similar('conflict', topn=25)
sim_words_17 = word2vec.wv.most_similar('terrorism', topn=25)
sim_words_18 = word2vec.wv.most_similar('education', topn=25)
print("Words most similar to woman are: " + str(sim_words))
print("Words most similar to man are: " + str(sim_words_2))
print("Words most similar to girl are: " + str(sim_words_3))
print("Words most similar to boy are: " + str(sim_words_4))
print("Words most similar to female are: " + str(sim_words_5))
print("Words most similar to male are: " + str(sim_words_6))
#print("Words most similar to depression are: " + str(sim_words_7))
#print("Words most similar to anxiety are: " + str(sim_words_8))
print("Words most similar to violence are: " + str(sim_words_9))
print("Words most similar to suicide are: " + str(sim_words_10))
print("Words most similar to leader are: " + str(sim_words_11))
print("Words most similar to addict are: " + str(sim_words_12))
print("Words most similar to alcoholic are: " + str(sim_words_13))
print("Words most similar to crash are: " + str(sim_words_14))
print("Words most similar to degree are: " + str(sim_words_15))
print("Words most similar to conflict are: " + str(sim_words_16))
print("Words most similar to terrorism are: " + str(sim_words_17))
print("Words most similar to education are: " + str(sim_words_18))