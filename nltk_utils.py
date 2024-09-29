import nltk
import numpy as np
#nltk.download('punkt_tab') # Make sure to download this package if you receive an error when running the program, pre-trained tokenizer
# the above package has pre-trained modules for various tasks in NLP

# importing a stemmer for stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())  # Stemming our words after the tokenizationa and converting them into lower case


def bag_of_words(tokenized_sentence, all_words):
    """ Tokenized sentence will be inputed here"""

    # Applying the stemming functionality to our tokenized sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence] 

    bag = np.zeros(len(all_words), dtype=np.float32) # initializing the bag array with zeros using numpy
    for idx, w in enumerate(all_words): # enumerate gives both the index and the current word
        if w in tokenized_sentence: # checking if the word is in our tokenized sentence
            bag[idx] = 1.0          # if it is, return 1.0

    return bag

