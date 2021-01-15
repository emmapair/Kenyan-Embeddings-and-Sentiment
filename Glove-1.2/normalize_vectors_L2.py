import numpy as np
import os
import csv
import re

##code from Github repository nikhgarg/EmbeddingDynamicStereotypes/dataset_utilities/normalize_vectors.py

def load_vectors(filename):
	vectors = {}
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter = ' ')
		for row in reader:
			word = re.sub('[^a-z]+', '', row[0].strip().lower())
			if len(word) < 2: continue
			vectors[word] = [float(x) for x in row[1:] if len(x) >0]
	return vectors

def print_sizes():
    #filenames_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_2000_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2001_2003_vectors_check_2.txt","glove_oped_lifestyle_news_business_counties_sports_2004_2006_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2007_2009_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2010_2012_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2013_2015_vectors_check_2.txt","glove_oped_lifestyle_news_business_counties_sports_2016_2018_vectors_check_2.txt"]
    filenames_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_1999_vectors_check_2.txt","glove_oped_lifestyle_news_business_counties_sports_2000_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2001_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2002_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2003_vectors_check_2.txt","glove_oped_lifestyle_news_business_counties_sports_2004_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2005_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2006_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2007_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2008_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2009_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2010_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2011_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2012_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2013_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2014_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2015_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2016_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2017_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2018_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2019_vectors_check_2.txt"]
    for name in filenames_glove:
        vectors = load_vectors(name)
        norms = [np.linalg.norm(vectors[word]) for word in vectors]
        print(np.mean(norms))
        print(np.var(norms))
        print(np.median(norms))

def normalize(filename, filename_output):
	vectors = {}
	countnorm0 = 0
	countnormal = 0
	with open(filename_output, 'w') as fo:
		writer = csv.writer(fo, delimiter = ' ')
		with open(filename, 'r') as f:
			reader = csv.reader(f, delimiter = ' ')
			for row in reader:
				rowout = row
				word = re.sub('[^a-z]+', '', row[0].strip().lower())
				rowout[0] = word
				if len(word) < 2: continue
				# print(word)
				norm = np.linalg.norm([float(x) for x in row[1:] if len(x) >0])
				if norm < 1e-2:
					countnorm0+=1
				else:
					countnormal+=1
					for en in range(1, len(rowout)):
						if len(rowout[en])>0:
							rowout[en] = float(rowout[en])/norm
					writer.writerow(rowout)
		fo.flush()
	print(countnorm0, countnormal)

def normalize_vectors():
	#filenames_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_2000_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2001_2003_vectors_check_2.txt","glove_oped_lifestyle_news_business_counties_sports_2004_2006_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2007_2009_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2010_2012_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2013_2015_vectors_check_2.txt","glove_oped_lifestyle_news_business_counties_sports_2016_2018_vectors_check_2.txt"]
	filenames_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_1999_vectors_check_2.txt","glove_oped_lifestyle_news_business_counties_sports_2000_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2001_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2002_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2003_vectors_check_2.txt","glove_oped_lifestyle_news_business_counties_sports_2004_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2005_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2006_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2007_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2008_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2009_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2010_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2011_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2012_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2013_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2014_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2015_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2016_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2017_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2018_vectors_check_2.txt", "glove_oped_lifestyle_news_business_counties_sports_2019_vectors_check_2.txt"]
	for name in filenames_glove:
		filename_output = name.replace('_vectors_check_2.txt','_vectors_normalized_check_2')
		print (name,filename_output)
		normalize(name, filename_output)

if __name__ == "__main__":
	normalize_vectors()          
