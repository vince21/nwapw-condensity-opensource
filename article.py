import pandas as pd
import nltk
from nltk import StanfordTagger
from nltk import word_tokenize
from nltk import sent_tokenize

class WordDataFrame:

    '''
        concatenates words from df into a string
        inputs: self, row number (int)
        outputs: sentence (string)
    '''
    def getSentenceByRow(self, row):
        return sent_tokenize(self.sentences[row])[0]


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
        fullText = ""

        with open(text) as f:
            for line in f:
                fullText += line

        self.sentences = sent_tokenize(fullText)

        wordData = []
        for sentence in self.sentences:
            wordData.append(word_tokenize(sentence))

        self.words = pd.DataFrame(wordData)


obj = WordDataFrame('test.txt')

print(obj.getTagsByRow(10))
print(obj.getNounsByRow(10))
print(obj.getVerbsByRow(10))
