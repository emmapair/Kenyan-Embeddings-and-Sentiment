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
    countlist = []
    validwords = []
    for word in wordlist:
        if word in vocab:
            count = vocab[word]
            countlist.append(vocab[word])
            print(vocab[word], word)
            if count > 49:
                validwords.append(word)
        else:
            print(word)
    print(countlist)
    return validwords
    
def calc_distance_between_vectors(vec1, vec2, distype):
    if distype is 'norm':
        return np.linalg.norm(np.subtract(vec1, vec2))
    else:
        return cossim(vec1, vec2)

##gender wordlists
female = ["she", "daughter", "hers", "her", 'mother', 'woman', 'girl', 'herself', 'female', 'sister', 'daughters', 'mothers', 'women', 'girls', 'females', 'sisters', 'aunt', 'aunts', 'niece', "nieces"]
#female = ['anne', 'waiguru', 'martha', 'karua', 'charity', 'ngilu', 'nancy', 'baraza', 'ann', 'ngirita', 'philomena', 'mwilu', 'gladys', 'shollei', 'susan', 'kihika', 'orie', 'rogo', 'manduli', 'esther', 'muthoni', 'passaris', 'margaret', 'wanjiru', 'millie', 'odhiambo', 'racheal', 'ruto', 'njoki', 'ndungu', 'gladys', 'wanga']
male = ["he", 'son', 'his', 'him', 'father', 'man', 'boy', 'himself', 'male', 'brother', 'sons', 'fathers', 'men', 'boys', 'males', 'brothers', 'uncle', 'uncles', 'nephew', 'nephews']
#male = ['uhuru', 'kenyatta', 'daniel', 'toroitich', 'arap', 'moi', 'william', 'ruto', 'raila', 'odinga', 'mwai', 'kibaki', 'kalonzo', 'musyoka', 'moses', 'wetangula', 'aden', 'duale', 'musalia', 'mudavadi', 'mike', 'mbuvi', 'sonko', 'evans', 'kidero', 'gideon', 'james', 'orengo', 'john', 'michuki', 'kiraitu', 'muriungi']

##leader nouns wordlist
#neutrallist = ['president', 'presidents', 'minister', 'ministers', 'leader', 'leaders', 'leadership', 'director', 'directors', 'officer', 'officers', 'chief', 'chiefs', 'authority', 'authorities', 'executive', 'executives', 'manager', 'managers', 'boss', 'bosses', 'politician', 'politicians', 'mayor', 'mayors', 'captain', 'captains', 'premier', 'premiers', 'governor', 'governors', 'commander', 'commanders', 'supervisor', 'supervisors']

#neutral wordlists
#neutrallist = ['competent', 'qualified','knowledgeable', 'accomplished', 'proficient', 'skilled', 'adept', 'practiced', 'experienced', 'expert', 'capable', 'able', 'skillful', 'prepared', 'credible', 'suitable', 'efficient', 'resourceful', 'educated', 'informed', 'professional', 'trained', 'talented', 'gifted', 'masterly', 'equipped', 'suited', 'masterful', 'pro', 'fit', 'ready', 'ace', 'master', 'schooled', 'overqualified', 'licensed', 'adequate', 'versed', 'employable', 'virtuoso', 'informed', 'erudite']
#neutrallist = ['intelligent', 'smart', 'clever', 'bright', 'shrewd', 'wise', 'sharp', 'genius', 'brilliant', 'discerning', 'canny', 'resourceful', 'intuitive', 'insightful', 'savvy', 'sagacious', 'adaptable', 'astute', 'keen', 'brainy', 'inventive', 'creative', 'imaginative', 'perceptive', 'intellectual', 'learned', 'observant', 'crafty', 'witty', 'educated', 'highbrow', 'original', 'ingenious', 'artful', 'sage', 'agile', 'quick', 'innovative', 'alert', 'cerebral', 'foresight']
#neutrallist = ['reasonable', 'sensible', 'rational', 'responsible', 'practical', 'strategic', 'diplomatic', 'calm', 'logical', 'agreeable', 'composed', 'prudent', 'pragmatic', 'realistic', 'reasoning', 'sane', 'calculating', 'politic', 'proper', 'commonsense', 'stable', 'mindful', 'cool', 'sober', 'grounded', 'analytical', 'judicious', 'tactical', 'tactful', 'levelheaded', 'aware', 'lucid', 'clearheaded', 'cautious', 'utilitarian', 'opportunistic', 'sapient', 'sound', 'analytic', 'justified', 'civil']
#neutrallist = ['influential', 'powerful', 'power', 'influence', 'confident', 'confidence', 'visionary', 'vision', 'strong', 'inspiring', 'engaging', 'charismatic', 'magnetic', 'respected', 'inspirational', 'esteemed', 'important', 'prominent', 'potent', 'significant', 'dominant', 'authoritative', 'leading', 'impressive', 'persuasive', 'forceful', 'eminent', 'notable', 'effective', 'charisma', 'famous', 'compelling', 'mighty', 'distinguished', 'famed', 'renowned', 'robust', 'convincing', 'foremost', 'commanding']
#neutrallist = ['considerate', 'virtuous', 'courteous', 'good', 'altruistic', 'integrity', 'trustworthy', 'accountable', 'transparent', 'humble', 'empathetic', 'positive', 'courageous', 'optimistic', 'bold', 'loyal', 'selfless', 'truthful', 'genuine', 'principled', 'best', 'straightforward', 'honorable', 'honourable', 'fair', 'sincere', 'moral', 'kind', 'benevolent', 'respectable', 'just', 'upright', 'honest', 'ethical', 'noble', 'thoughtful', 'respectful', 'excellent', 'upstanding', 'righteous', 'responsible', 'heroic', 'brave', 'valiant']
#neutrallist = ['determined', 'dedicated', 'committed', 'purposeful', 'persevering', 'ambitious', 'devoted', 'unwavering', 'steadfast', 'caring', 'faithful', 'diligent', 'industrious', 'steady', 'resolute', 'firm', 'decided', 'tenacious', 'resolved', 'unshakeable', 'persistent', 'certain', 'tireless', 'spirited', 'decisive', 'constant', 'dogged', 'intentional', 'serious', 'staunch', 'aspiring', 'enterprising', 'focused', 'solid', 'intent', 'devout', 'dutiful', 'unfaltering', 'patient', 'assured']
#neutrallist = ['loose', 'immoral', 'prostitute', 'indecent', 'lewd', 'lustful', 'promiscuous', 'wanton', 'racy', 'immodest', 'dirty', 'careless', 'nasty', 'inappropriate', 'filthy', 'improper', 'vulgar', 'sinful', 'wicked', 'depraved', 'vile', 'shameless', 'impure', 'shameful', 'lecherous', 'ungodly', 'carnal', 'obscene', 'raunchy', 'crude', 'provocative', 'scandalous', 'sleazy', 'wretched', 'trashy', 'profane', 'raunchy', 'bawdy', 'foul', 'smutty']
#neutrallist = ['homemaker', 'domestic', 'rearing', 'childrearing', 'housekeeping', 'parenting', 'cleaning', 'childcare', 'household', 'caregiving', 'errands', 'children', 'caregiver', 'babysitting', 'home', 'laundry', 'cooking', 'washing', 'caretaker', 'kitchen', 'housework', 'family', 'chores', 'house', 'scrubbing', 'tidying', 'sweeping', 'scrub', 'sweep', 'cook', 'babysit', 'wash', 'child', 'chore', 'dusting', 'ironing', 'mopping', 'mop', 'iron', 'sewing', 'sew', 'schoolwork', 'errand', 'housekeeper', 'maid']
#neutrallist = ['corruption', 'duplicity', 'corrupted', 'immoral', 'bribe', 'bribes', 'bribed', 'bribery', 'fraud', 'fraudulent', 'manipulate', 'corrupt', 'nefarious', 'shady', 'unethical', 'crooked', 'untrustworthy', 'exploit', 'dishonest', 'dishonesty', 'crime', 'graft', 'extortion', 'nepotism', 'theft', 'misappropriation', 'thief', 'irregularities', 'irregular', 'fictitious', 'wrongdoing', 'robbery', 'crookedness', 'injustice', 'exploitation', 'criminal', 'criminality', 'unlawful', 'illegitimate', 'felony', 'felonies', 'debauchery', 'crime']
#neutrallist = ['authoritarian', 'tyrannical', 'tyranny', 'tyrant', 'cruel', 'cruelty', 'dictatorial', 'dictator', 'dictatorship', 'autocratic', 'totalitarian', 'repressive', 'ruthless', 'oppressive', 'undemocratic', 'corrupt', 'tribal', 'violent', 'violence', 'rude', 'inciter', 'chaos', 'chaotic', 'difficult', 'cronyism', 'bossy', 'oppressor', 'bully', 'evil', 'warlord', 'inhumane', 'brutal', 'hateful', 'malevolent', 'sadistic', 'savage', 'hostile', 'instigator', 'despot', 'despotic']
#neutrallist = ['scandal', 'slander', 'libel', 'defamed', 'smear', 'stigma', 'discredited', 'mudslinging', 'vilification', 'insult', 'defamation', 'allegations', 'scam', 'tarnished', 'criticized', 'criticism', 'rumour', 'calumny', 'disparagement', 'defame', 'insulted', 'defamatory', 'discredit', 'gossip', 'belittlement', 'dirt', 'accusation', 'accusations', 'dishonour', 'debasement', 'shame', 'disparaged', 'vilified', 'denigrated', 'malign', 'allegation', 'dishonoured', 'debased', 'shamed', 'demeaned']
neutrallist = ['homemaker', 'domestic', 'rearing', 'childrearing', 'housekeeping', 'parenting', 'cleaning', 'childcare', 'household', 'caregiving', 'errands', 'children', 'caregiver', 'babysitting', 'home', 'laundry', 'cooking', 'washing', 'caretaker', 'kitchen', 'housework', 'family', 'chores', 'house', 'scrubbing', 'tidying', 'sweeping', 'scrub', 'sweep', 'cook', 'babysit', 'wash', 'child', 'chore', 'dusting', 'ironing', 'mopping', 'mop', 'iron', 'sewing', 'sew', 'schoolwork', 'errand', 'housekeeper', 'maid', 'househelp']

biasvector = []
avg1vector = []
avg2vector = []

##for 3 year vectors
filenames_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_2000_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2001_2003_vectors_normalized_check_2","glove_oped_lifestyle_news_business_counties_sports_2004_2006_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2007_2009_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2010_2012_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2013_2015_vectors_normalized_check_2","glove_oped_lifestyle_news_business_counties_sports_2016_2018_vectors_normalized_check_2"]
vocab_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_2000_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2001_2003_vocab_check_2","glove_oped_lifestyle_news_business_counties_sports_2004_2006_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2007_2009_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2010_2012_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2013_2015_vocab_check_2","glove_oped_lifestyle_news_business_counties_sports_2016_2018_vocab_check_2"]

##for 1 year vectors
#filenames_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_1999_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2000_vectors_normalized_check_2","glove_oped_lifestyle_news_business_counties_sports_2001_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2002_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2003_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2004_vectors_normalized_check_2","glove_oped_lifestyle_news_business_counties_sports_2005_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2006_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2007_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2008_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2009_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2010_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2011_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2012_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2013_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2014_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2015_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2016_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2017_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2018_vectors_normalized_check_2", "glove_oped_lifestyle_news_business_counties_sports_2019_vectors_normalized_check_2"]
#vocab_glove = ["glove_oped_lifestyle_news_business_counties_sports_1998_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_1999_vocab_check_2","glove_oped_lifestyle_news_business_counties_sports_2000_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2001_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2002_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2003_vocab_check_2","glove_oped_lifestyle_news_business_counties_sports_2004_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2005_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2006_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2007_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2008_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2009_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2010_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2011_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2012_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2013_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2014_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2015_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2016_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2017_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2018_vocab_check_2", "glove_oped_lifestyle_news_business_counties_sports_2019_vocab_check_2"]


for i in range(0,7):
#for i in range(0,22):
    vectors = load_vectors(filenames_glove[i])
    vocab = load_vocab(vocab_glove[i])
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
    #avg1=[calc_distance_between_vectors(average_vector_1,vectors[word], distype = 'cos') for word in validwords3]
    #avg2=[calc_distance_between_vectors(average_vector_2,vectors[word], distype = 'cos') for word in validwords3]

    print("avg1", avg1)
    avg1vector.append(avg1)
    print("avg2", avg2)
    avg2vector.append(avg2)

    #Step 3: Did this for 2 target word lists, then subtracted one from the other.
    bias = avg1-avg2
    print(bias)
    biasvector.append(bias)

print(avg1vector)
print(avg2vector)
print(biasvector)
                        
