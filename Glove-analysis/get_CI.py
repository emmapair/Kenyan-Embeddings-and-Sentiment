import numpy as np
import os
import csv
import re

def load_vocab(fi):
    try:
        with open(fi, 'r') as f:
            reader = csv.reader(f, delimiter = ' ')
            return {d[0]:float(d[1]) for d in reader}
    except:
        return None
        
def load_vectors(filename):
    print (filename)
    vectors = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter = ' ')
        for row in reader:
            vectors[row[0]] = [float(x) for x in row[1:] if len(x) >0]
    return vectors
    
def cossim(v1, v2, signed = True):
    c = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    if not signed:
        return abs(c)
    return c
        
def bootstrap(data1, data2, n=10000, func=np.mean):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    simulations = list()
    sample_size_1 = len(data1)
    xbar_init_1 = np.mean(data1)
    sample_size_2 = len(data2)
    xbar_init_2 = np.mean(data2)
    for c in range(n):
        itersample1 = np.random.choice(data1, size=sample_size_1, replace=True)
        itersample2 = np.random.choice(data2, size=sample_size_2, replace=True)
        simulations.append(func(itersample1)-func(itersample2))
    simulations.sort()
    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified
        by p.
        """
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(n*l_pval))
        u_indx = int(np.floor(n*u_pval))
        return(simulations[l_indx],simulations[u_indx])
    return(ci)
    
vectors = ()
vocab = ()

def validwords(wordlist):
    validwords = []
    for word in wordlist:
        if word in vocab:
            count = vocab[word]
            print(vocab[word], word)
            if count > 49:
                validwords.append(word)
        else:
            print(word)
    return validwords
    
def calc_distance_between_vectors(vec1, vec2, distype):
    if distype is 'norm':
        return np.linalg.norm(np.subtract(vec1, vec2))
    else:
        return cossim(vec1, vec2)
    
female = ["she", "daughter", "hers", "her", 'mother', 'woman', 'girl', 'herself', 'female', 'sister', 'daughters', 'mothers', 'women', 'girls', 'females', 'sisters', 'aunt', 'aunts', 'niece', "nieces"]
male = ["he", 'son', 'his', 'him', 'father', 'man', 'boy', 'himself', 'male', 'brother', 'sons', 'fathers', 'men', 'boys', 'males', 'brothers', 'uncle', 'uncles', 'nephew', 'nephews']
neutrallist = ['president', 'presidents', 'minister', 'ministers', 'leader', 'leaders', 'leadership', 'director', 'directors', 'officer', 'officers', 'chief', 'chiefs', 'authority', 'authorities','executive', 'executives', 'manager', 'managers', 'boss', 'bosses', 'politician', 'politicians', 'mayor', 'mayors', 'captain', 'captains', 'premier', 'premiers', 'governor', 'governors', 'commander', 'commanders', 'supervisor', 'supervisors']

#filenames_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_2000_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2001_2003_vectors_normalized_check_2","glove_oped_lifestyle_news_business_counties_sports_2004_2006_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2007_2009_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2010_2012_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2013_2015_vectors_normalized_check_2","glove_oped_lifestyle_news_business_counties_sports_2016_2018_vectors_normalized_check_2"]
#vocab_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_2000_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2001_2003_vocab_check_2","glove_oped_lifestyle_news_business_counties_sports_2004_2006_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2007_2009_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2010_2012_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2013_2015_vocab_check_2","glove_oped_lifestyle_news_business_counties_sports_2016_2018_vocab_check_2"]

filenames_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_1999_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2000_vectors_normalized_check_2","glove_oped_lifestyle_news_business_counties_sports_2001_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2002_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2003_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2004_vectors_normalized_check_2","glove_oped_lifestyle_news_business_counties_sports_2005_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2006_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2007_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2008_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2009_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2010_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2011_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2012_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2013_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2014_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2015_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2016_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2017_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2018_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2019_vectors_normalized_check_2"]
vocab_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_1999_vocab_check_2","glove_oped_lifestyle_news_business_counties_sports_2000_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2001_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2002_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2003_vocab_check_2","glove_oped_lifestyle_news_business_counties_sports_2004_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2005_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2006_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2007_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2008_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2009_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2010_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2011_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2012_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2013_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2014_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2015_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2016_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2017_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2018_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2019_vocab_check_2"]

#for i in range(0,7):
for i in range(0,22):
    vectors = load_vectors(filenames_glove[i])
    vocab = load_vocab(vocab_glove[i])

    validwords1 = validwords(female)
    validwords2 = validwords(male)
    validwords3 = validwords(neutrallist)

    average_vector_1 = np.mean(np.array([vectors[word] for word in validwords1]), axis = 0)
    average_vector_2 = np.mean(np.array([vectors[word] for word in validwords2]), axis = 0)
    #print(average_vector_1)

    v1 = ([calc_distance_between_vectors(average_vector_1,vectors[word], distype = 'cos') for word in validwords3])
    v2 = ([calc_distance_between_vectors(average_vector_2,vectors[word], distype = 'cos') for word in validwords3])

    boot = bootstrap(v1, v2)
    cinterval = [boot(.95)]
    print(cinterval)
    #cintervals1 = [boot1(i) for i in (.90, .95, .99, .995)]
    #print(cintervals1)


















