# Kenyan-Embeddings-and-Sentiment

This file overviews the steps the authors took in the gender bias analysis of word embeddings and sentiment in the Daily Nation, a Kenyan newspaper. This repository does not contain the original text files of uncleaned text from the Daily Nation, the word embeddings (both GloVe and word2vec), word embedding vocabulary files, or the sentence files used for the sentiment analysis. These are held in Harvard Dataverse and can be accessed at this link: https://dataverse.harvard.edu/dataverse/kenyan-embeddings-and-sentiment

This Github repository contains the code used for the gender bias analysis of the embeddings and sentences files.

Much of this code is from another Github repo created by a fellow researcher also measuring gender bias in word embeddings. That repo can be accessed here: https://github.com/nikhgarg/EmbeddingDynamicStereotypes

# Word2vec Word Embeddings Analysis Steps
1)	Use the word2vec_combine_clean.py to combine and clean the raw Daily Nation text files by the years and subjects you want combined and create the word2vec word embeddings from this combined text. Again, the uncleaned text files are held in Harvard Dataverse.
2)	Use the normalize_vectors_L2.py to normalize the vectors so that they can be used to measure the gender bias in the embeddings.
3)	To measure gender bias in the embeddings use bias_word2vec.py.
4)	To create a text file format that can be read into the GloVe algorithm use glove_combine_clean.py to combine and clean the Daily Nation text files by the years and subjects you want combined.

# Glove Word Embeddings Analysis Steps
1)	“Glove-1.2” contains the code from the original GloVe Github repository (https://github.com/stanfordnlp/GloVe). The demo.sh file can be edited to create word embeddings from a text file (CORPUS) and a vocab file name can be specified (VOCAB_FILE).
2)	Once the GloVe embeddings have been created, use normalize_vectors_L2.py to normalize the vectors so that they can be used to measure the gender bias in the embeddings.
3)	In the “Glove-analysis” folder you can measure the gender bias in the embeddings using bias_glove.py.
4)	If you want to get a confidence interval for the bias in the GloVe embeddings use get_CI.py.

# Sentiment Analysis Steps
1)	Use get_sentences.py to create sentences using the original uncleaned text files. We pulled sentences containing the names of male and female political leaders and the two surrounding sentences.
2)	Use Sentiment Analysis Final.ipynb to calculate the sentiment of these sentences.
