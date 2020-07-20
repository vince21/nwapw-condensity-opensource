import pandas as pd
import nltk
from nltk import StanfordTagger
from nltk import word_tokenize

class WordDataFrame:

    def getSentenceByRow(self, row):
        sentence = ""
        for word in self.df.iloc[row]:
            if not word is None:
                sentence += str(word) + " "
        return sentence

    def getTagsByRow(self, row):
        verbs = []
        string = "my name is vincent. I like to eat cake."
        tagged = nltk.pos_tag(nltk.word_tokenize(self.getSentenceByRow(row)))
        return tagged



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
print(obj.getTagsByRow(2))
