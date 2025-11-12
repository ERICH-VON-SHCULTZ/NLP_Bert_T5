import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    
    # --- Random Synonym Replacement ---
    text = example["text"]
    words = word_tokenize(text)
    new_words = []
    
    # Set the probability of replacing a word
    replacement_prob = 0.30 

    for word in words:
        # Check if we should replace this word based on probability
        if random.random() < replacement_prob:
            synonyms = set()
            
            # Find all synonyms using WordNet
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    # Get the synonym name
                    synonym = lemma.name()
                    # Add it to the set if it's different from the original word
                    # and doesn't contain underscores (which WordNet uses for phrases)
                    if synonym.lower() != word.lower() and '_' not in synonym:
                        synonyms.add(synonym)
            
            # If we found at least one valid synonym, pick one randomly
            if len(synonyms) > 0:
                new_word = random.choice(list(synonyms))
                new_words.append(new_word)
            else:
                # No valid synonyms found, just append the original word
                new_words.append(word)
        else:
            # We decided not to replace this word, append the original
            new_words.append(word)
            
    # Detokenize the list of words back into a single string
    new_text = TreebankWordDetokenizer().detokenize(new_words)
    
    example["text"] = new_text

    ##### YOUR CODE ENDS HERE ######

    return example
