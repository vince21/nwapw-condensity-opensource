import pandas as pd
import nltk
from nltk import StanfordTagger
from nltk import word_tokenize

class WordDataFrame:

    '''
        concatenates words from df into a strings
        inputs: self, row number (int)
        outputs: sentence (string)
    '''
    def getSentenceByRow(self, row):
        sentence = ""
        for word in self.df.iloc[row]:
            if not word is None:
                sentence += str(word) + " "
        return sentence


    '''
        tags words by part of speech
        inputs: self, row number (int)
        outputs:  words and their part of speech (array[tuples])
    '''
    def getTagsByRow(self, row):
        tagged = nltk.pos_tag(nltk.word_tokenize(self.getSentenceByRow(row)))
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
        sentenceData = []

        with open(text) as f:
            for line in f:
                sentenceData.append(line.split('.')[:-1])

        wordData = []
        for sentence in sentenceData:
            for words in sentence:
                wordData.append(words.split(' '))

        self.df = pd.DataFrame(wordData)

obj = WordDataFrame('test.txt')
#print(obj.getSentenceByRow(2))
print(obj.getNounsByRow(3))
