import nltk
from nltk import StanfordTagger
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd


class WordDataFrame:

    '''
        tags words by part of speech
        inputs: self, row number (int)
        outputs:  words and their part of speech (array[tuples])
    '''
    def getTagsByRow(self, row):

        tagged = nltk.pos_tag(nltk.word_tokenize(self.sentences[row]))
        return tagged

    '''
        finds verbs in a sentence
        inputs: self, row number (int)
        outputs:  words and their part of speech (array[VERBS])
    '''
    def getVerbsByRow(self, row):
        verbs = []
        tagged = self.getTagsByRow(row)
        for word in tagged:
            if word[1] == "VB":
                verbs.append(word[0])
        return verbs

    '''
        finds nouns in a sentence
        inputs: self, row number (int)
        outputs:  words and their part of speech (array[Nouns])
    '''
    def getNounsByRow(self, row):
        nouns = []
        tagged = self.getTagsByRow(row)
        for word in tagged:
            if word[1] == "NN":
                nouns.append(word[0])
        return nouns

    '''
        Constructor
        inputs: self, text (.txt file) //TODO: we will want to chance this so that links or raw text can be inputted
    '''
    def __init__(self, text):
        wnl = WordNetLemmatizer()
        fullText = ""

        with open(text) as f:
            for line in f:
                fullText += line

        self.sentences = sent_tokenize(fullText)

        wordData = []
        lemmas = []
        for sentence in self.sentences:
            words = word_tokenize(sentence)
            wordData.append(words)
            for word in words:
                if word not in lemmas:
                    lemmas.append(wnl.lemmatize(word))



        self.words = wordData
        self.wordDF = pd.DataFrame(lemmas)

        self.sentences_df = pd.DataFrame.from_dict({'sentences': self.sentences})

        # temp var "words"— can replace with list of lemmatized words; list comp flattens self.words
        words = list(set([word for sentence in self.words for word in sentence]))
        self.words_df = pd.DataFrame.from_dict({'words': words})


obj = WordDataFrame('test.txt')

print(obj.wordDF)
print(obj.getNounsByRow(10))
print(obj.getVerbsByRow(10))
