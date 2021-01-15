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

startyr = 2016
endyr = 2018
jump = 1
subject = ["oped","lifestyle","news", "business", "counties", "sports"]
title = "oped_lifestyle_news_business_counties_sports"
out_filename = "daily_nation_" + str(title) + "_" + str(startyr) + "_" + str(endyr) + "_combined.txt"
#out_filename = "daily_nation_" + str(title) + "_" + str(startyr) + "_combined.txt"
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
#print(sentences[:50])

new_sentences_male2 = [];
new_sentences_female2 = [];
new_sentences_male1 = [];
new_sentences_female1 = [];
female1 = ["she", "daughter", "hers", "her", 'mother', 'woman', 'girl', 'herself', 'female', 'sister', 'daughters', 'mothers', 'women', 'girls', 'females', 'sisters', 'aunt', 'aunts', 'niece', "nieces"]
female2 = ['anne', 'waiguru', 'martha', 'karua', 'charity', 'ngilu', 'nancy', 'baraza', 'ann', 'ngirita', 'philomena', 'mwilu', 'gladys', 'shollei', 'susan', 'kihika', 'orie', 'rogo', 'manduli', 'esther', 'muthoni', 'passaris', 'margaret', 'wanjiru', 'millie', 'odhiambo', 'racheal', 'ruto', 'njoki', 'ndungu', 'gladys', 'wanga']
male1 = ["he", 'son', 'his', 'him', 'father', 'man', 'boy', 'himself', 'male', 'brother', 'sons', 'fathers', 'men', 'boys', 'males', 'brothers', 'uncle', 'uncles', 'nephew', 'nephews']
#male2 = ['uhuru', 'kenyatta', 'daniel', 'toroitich', 'arap', 'moi', 'william', 'ruto', 'raila', 'odinga', 'mwai', 'kibaki', 'kalonzo', 'musyoka', 'moses', 'wetangula', 'aden', 'duale', 'musalia', 'mudavadi', 'mike', 'mbuvi', 'sonko']
male2 = ['uhuru', 'kenyatta', 'daniel', 'toroitich', 'arap', 'moi', 'william', 'ruto', 'raila', 'odinga', 'mwai', 'kibaki', 'kalonzo', 'musyoka', 'moses', 'wetangula', 'aden', 'duale', 'musalia', 'mudavadi', 'mike', 'mbuvi', 'sonko', 'evans', 'kidero', 'gideon', 'james', 'orengo', 'john', 'michuki', 'kiraitu', 'muriungi']
leaders = ['president', 'presidents', 'minister', 'ministers', 'leader', 'leaders', 'leadership', 'director', 'directors', 'officer', 'officers', 'chief', 'chiefs', 'authority', 'authorities', 'executive', 'executives', 'manager', 'managers', 'boss', 'bosses', 'politician', 'politicians', 'mayor', 'mayors', 'captain', 'captains', 'premier', 'premiers', 'governor', 'governors', 'commander', 'commanders', 'supervisor', 'supervisors']

for i in range(len(sentences)): 
    for word in male2:
        if word in sentences[i] and sentences[i] not in new_sentences_male2:
            new_sentences_male2.append(sentences[i])
            if i < len(sentences)-1 and i != 0:
                if sentences[i-1] not in new_sentences_male2:
                    new_sentences_male2.append(sentences[i-1])
                if sentences[i+1] not in new_sentences_male2:
                    new_sentences_male2.append(sentences[i+1])
                
#for i in range(len(sentences)): 
#    for word in female2:
#        if word in sentences[i] and sentences[i] not in new_sentences_female2:
#            new_sentences_female2.append(sentences[i])
#            if i < len(sentences)-1 and i != 0:
#                if sentences[i-1] not in new_sentences_female2:
#                    new_sentences_female2.append(sentences[i-1])
#                if sentences[i+1] not in new_sentences_female2:
#                    new_sentences_female2.append(sentences[i+1])
                
#for i in range(len(sentences)): 
#    for word in male1:
#        for w in leaders:
#            if word in sentences[i] and w in sentences[i] and sentences[i] not in new_sentences_male1:
#                new_sentences_male1.append(sentences[i])
#                if i < len(sentences)-1 and i != 0:
#                    if sentences[i-1] not in new_sentences_male1:
#                        new_sentences_male1.append(sentences[i-1])
#                    if sentences[i+1] not in new_sentences_male1:
#                        new_sentences_male1.append(sentences[i+1])
                
#for i in range(len(sentences)): 
#    for word in female1:
#        for w in leaders:
#            if word in sentences[i] and w in sentences[i] and sentences[i] not in new_sentences_female1:
#                new_sentences_female1.append(sentences[i])
#                if i < len(sentences)-1 and i != 0:
#                    if sentences[i-1] not in new_sentences_female1:
#                        new_sentences_female1.append(sentences[i-1])
#                    if sentences[i+1] not in new_sentences_female1:
#                        new_sentences_female1.append(sentences[i+1])

## Creating a cleaned output file
print ("Creating cleaned text file...")
out_cleaned_filename_male2 = "daily_nation_" + str(title) + "_" + str(startyr) +"_" + str(endyr) + "_sentences_male2_2.txt" # Name output file
f = open(out_cleaned_filename_male2, "w")
f.write(str(new_sentences_male2))
f.close()

#out_cleaned_filename_female2 = "daily_nation_" + str(title) + "_" + str(startyr) +"_" + str(endyr) + "_sentences_female2.txt" # Name output file
#f = open(out_cleaned_filename_female2, "w")
#f.write(str(new_sentences_female2))
#f.close()

#out_cleaned_filename_male1 = "daily_nation_" + str(title) + "_" + str(startyr) +"_" + str(endyr) + "_sentences_male1.txt" # Name output file
#f = open(out_cleaned_filename_male1, "w")
#f.write(str(new_sentences_male1))
#f.close()

#out_cleaned_filename_female1 = "daily_nation_" + str(title) + "_" + str(startyr) +"_" + str(endyr) + "_sentences_female1.txt" # Name output file
#f = open(out_cleaned_filename_female1, "w")
#f.write(str(new_sentences_female1))
#f.close()