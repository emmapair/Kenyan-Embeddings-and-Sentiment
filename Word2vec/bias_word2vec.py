import csv
import numpy as np
import sys
import copy
import datetime

def load_vectors(filename):
    print (filename)
    vectors = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter = ' ')
        for row in reader:
            vectors[row[0]] = [float(x) for x in row[1:] if len(x) >0]
    return vectors
    
def load_vocab(fi):
    try:
        with open(fi, 'r') as f:
            reader = csv.reader(f, delimiter = ' ')
            return {d[0]:float(d[1]) for d in reader}
    except:
        return None
    
def cossim(v1, v2, signed = True):
    c = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    if not signed:
        return abs(c)
    return c
    
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
        
##Gender wordlists
#female = ["she", "daughter", "hers", "her", 'mother', 'woman', 'girl', 'herself', 'female', 'sister', 'daughters', 'mothers', 'women', 'girls', 'females', 'sisters', 'aunt', 'aunts', 'niece', "nieces"]
female = ['anne', 'waiguru', 'martha', 'karua', 'charity', 'ngilu', 'nancy', 'baraza', 'ann', 'ngirita', 'philomena', 'mwilu', 'gladys', 'shollei', 'susan', 'kihika', 'orie', 'rogo', 'manduli', 'esther', 'muthoni', 'passaris', 'margaret', 'wanjiru', 'millie', 'odhiambo', 'racheal', 'ruto', 'njoki', 'ndungu', 'gladys', 'wanga']
#male = ["he", 'son', 'his', 'him', 'father', 'man', 'boy', 'himself', 'male', 'brother', 'sons', 'fathers', 'men', 'boys', 'males', 'brothers', 'uncle', 'uncles', 'nephew', 'nephews']
male = ['uhuru', 'kenyatta', 'daniel', 'toroitich', 'arap', 'moi', 'william', 'ruto', 'raila', 'odinga', 'mwai', 'kibaki', 'kalonzo', 'musyoka', 'moses', 'wetangula', 'aden', 'duale', 'musalia', 'mudavadi', 'mike', 'mbuvi', 'sonko']

##Neutral wordlist
#neutrallist = ['president', 'presidents', 'minister', 'ministers', 'leader', 'leaders', 'leadership', 'director', 'directors', 'officer', 'officers', 'chief', 'chiefs', 'authority', 'authorities', 'executive', 'executives', 'manager', 'managers', 'boss', 'bosses', 'politician', 'politicians', 'mayor', 'mayors', 'captain', 'captains', 'premier', 'premiers', 'governor', 'governors', 'commander', 'commanders', 'supervisor', 'supervisors']

biasvector = []
avg1vector = []
avg2vector = []

#filenames_gensim = ["daily_nation_oped_lifestyle_news_business_counties_sports_1998_2000_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2001_2003_vectors_normalized_check.txt","daily_nation_oped_lifestyle_news_business_counties_sports_2004_2006_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2007_2009_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2010_2012_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2013_2015_vectors_normalized_check.txt","daily_nation_oped_lifestyle_news_business_counties_sports_2016_2018_vectors_normalized_check.txt"]
#vocab_gensim = ["daily_nation_oped_lifestyle_news_business_counties_sports_1998_2000_vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2001_2003_vocab_check.txt","daily_nation_oped_lifestyle_news_business_counties_sports_2004_2006_vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2007_2009_vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2010_2012_vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2013_2015_vocab_check.txt","daily_nation_oped_lifestyle_news_business_counties_sports_2016_2018_vocab_check.txt"]

filenames_gensim = ["daily_nation_oped_lifestyle_news_business_counties_sports_1998_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_1999_vectors_normalized_check.txt","daily_nation_oped_lifestyle_news_business_counties_sports_2000_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2001_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2002_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2003_vectors_normalized_check.txt","daily_nation_oped_lifestyle_news_business_counties_sports_2004_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2005_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2006_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2007_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2008_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2009_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2010_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2011_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2012_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2013_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2014_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2015_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2016_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2017_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2018_vectors_normalized_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2019_vectors_normalized_check.txt"]
vocab_gensim = ["daily_nation_oped_lifestyle_news_business_counties_sports_1998__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_1999__vocab_check.txt","daily_nation_oped_lifestyle_news_business_counties_sports_2000__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2001__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2002__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2003__vocab_check.txt","daily_nation_oped_lifestyle_news_business_counties_sports_2004__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2005__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2006__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2007__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2008__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2009__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2010__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2011__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2012__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2013__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2014__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2015__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2016__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2017__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2018__vocab_check.txt", "daily_nation_oped_lifestyle_news_business_counties_sports_2019__vocab_check.txt"]

for i in range(0,22):
    vectors = load_vectors(filenames_gensim[i])
    vocab = load_vocab(vocab_gensim[i])
    validwords1 = validwords(female)
    validwords2 = validwords(male)
    validwords3 = validwords(neutrallist)

    #Step 1: Create a vector for a target word list concept (e.g., gender) -- for each
    #word in the (neutral?) list, get its vector and then average the vectors.
    average_vector_1 = np.mean(np.array([vectors[word] for word in validwords1]), axis = 0)
    average_vector_2 = np.mean(np.array([vectors[word] for word in validwords2]), axis = 0)
    #print(average_vector_1)

    #seeing what words in the list are more associated with men and women
    dict1 = {}
    dict2 = {}
    for word in validwords3:
        distance1 = calc_distance_between_vectors(average_vector_1,vectors[word], distype = 'cos')
        distance2 = calc_distance_between_vectors(average_vector_2,vectors[word], distype = 'cos')
        dict1[word] = distance1
        dict2[word] = distance2
    sorted1 = sorted(dict1, key=dict1.__getitem__)
    sorted2 = sorted(dict2, key=dict2.__getitem__)
    print(sorted1)
    print(sorted2)
    
    #Step 2: For each of the neutral word list (like occupation), I got the distance of that vector
    #to the vector from the first step.
    avg1=np.mean([calc_distance_between_vectors(average_vector_1,vectors[word], distype = 'cos') for word in validwords3])
    avg2=np.mean([calc_distance_between_vectors(average_vector_2,vectors[word], distype = 'cos') for word in validwords3])

    print(avg1)
    avg1vector.append(avg1)
    print(avg2)
    avg2vector.append(avg2)

    #Step 3: Did this for 2 target word lists, then subtracted one from the other.
    bias = avg1-avg2
    print(bias)
    biasvector.append(bias)

print(avg1vector)
print(avg2vector)
print(biasvector)          
