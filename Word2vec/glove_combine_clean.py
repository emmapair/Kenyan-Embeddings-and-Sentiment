import subprocess
import os.path
import csv
import re
import random
import requests
from urllib.parse import urljoin
from urllib.request import urlopen as uReq
import re
import nltk
import gensim
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import string
import os.path
from os import path

##For three year word embeddings.
#startyr = 1998
#endyr = 2000

year = [1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
jump = 1
subject = ["oped","lifestyle","news", "business", "counties", "sports"]
title = "oped_lifestyle_news_business_counties_sports"
for yer in year:
    #out_filename = "glove_" + str(title) + "_" + str(startyr) + "_" + str(endyr) + "_combined.txt"
    out_filename = "glove_" + str(title) + "_" + str(yer) + "_combined.txt"
    print ("Creating combined text file...")
    with open(out_filename, 'w') as f:
        for heading in subject:
            for yr in range(yer, yer+1, jump):
                newfile = 'daily_nation_' + str(heading) + "_" + str(yer)+ '_uncleaned.txt'
                if path.exists(newfile):
                    with open(newfile, 'r', encoding="utf8") as fread:
                        print (str(yer))
                        for line in fread:
                            f.write(line)
    f.close()

    # Get text out of textfile
    file = open(out_filename, "r") # Choose textfile to clean
    articles = file.read()

    ###Preprocessing
    ### Cleaing the text
    print ("Changing to lowercase and removing nonalphabetical characters...")
    processed_article = str(articles).lower()
    #processed_article = re.sub('[^a-zA-Z\-]', ' ', processed_article )
    processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
    processed_article = re.sub(r'\s+', ' ', processed_article)
    #out_cleaned_filename = "glove_" + str(title) + "_" + str(startyr) +"_" + str(endyr) + "_precleaned.txt" # Name output file
    out_cleaned_filename = "glove_" + str(title) + "_" + str(yer) + "_cleaned.txt" # Name output file
    f = open(out_cleaned_filename, "w")
    f.write(str(processed_article))
    f.close()
    
    
    