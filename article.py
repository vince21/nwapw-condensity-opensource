import nltk
from nltk import StanfordTagger
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

    def get_sentiment(self):
        """
        gets the total sentiment of the text
        :return: tuple containing the average sentiment and the average nonzero sentiment
        :rtype: (float, float)
        """
        total_sentiment = self.sentencesDF['Sentiment'].mean()
        nonzero_sentiment = self.sentencesDF[self.sentencesDF['Sentiment'] != 0]['Sentiment'].mean()
        return total_sentiment, nonzero_sentiment

    def score_word(self, word):
        # placeholder
        # should contain algorithm that weights word frequency, etc.
        # should reference self.wordDF
        return 1

    def score_sentence(self, sentence):
        """
        Takes in a sentence and returns its score
        :param sentence: element of self.sentences
        :type sentence: str
        :return: score of sentence
        :rtype: float
        """
        words = word_tokenize(sentence)
        score = 0
        for word in words:
            score += self.score_word(word)
        score /= len(words)
        return score

    def condense(self, percent):
        """
        Condenses the text in self.sentences by a given percent
        :param percent: Specifies how much the text should be reduced by (0 < percent <= 1)
        :type percent: float
        :return: String containing abbreviated text
        :rtype: str
        """

        # scores each sentence based on score_sentence function
        sentence_scores = [(sentence, self.score_sentence(sentence)) for sentence in self.sentences]
        # sorts by best score
        sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        # calculates number of sentences to return based on input
        num_sentences = int(len(sentences) * percent)
        if num_sentences < 1:
            num_sentences = 1
        elif num_sentences > len(sentences):
            num_sentences = len(sentences)

        sentences_to_return = sentences[:num_sentences]

        # gets rid of tuples w/ score
        sentences_to_return = [sentence[0] for sentence in sentences_to_return]

        # sort sentences in original chronological order
        sentences_to_return.sort(key=lambda x: self.sentences.index(x))

        # joins sentences to make text body
        return ' '.join(sentences_to_return)

    '''
        Constructor
        inputs: self, text (.txt file) //TODO: we will want to chance this so that links or raw text can be inputted
    '''
    def __init__(self, text):
        wnl = WordNetLemmatizer()
        fullText = ""

        # check if text is file or not
        if text.split('.')[-1] == 'txt':
            with open(text) as f:
                for line in f:
                    fullText += line
        else:
            fullText = text

        self.sentences = sent_tokenize(fullText)


        wordData = []
        lemmas = []
        for sentence in self.sentences:
            words = word_tokenize(sentence)
            wordData.append(words)
            for word in words:
                lemmas.append(wnl.lemmatize(word))

        self.words = wordData
        self.wordDF = pd.DataFrame.from_dict({'Words': word_tokenize(fullText),'Lemmas': lemmas,'Scores': 0})

        # sentiment analysis for sentences
        sia = SentimentIntensityAnalyzer()
        sentence_sentiments = [sia.polarity_scores(sentence)['compound'] for sentence in self.sentences]
        self.sentencesDF = pd.DataFrame.from_dict({'Sentence': self.sentences,
                                                   'Sentiment': sentence_sentiments
                                                   })







obj = WordDataFrame('test.txt')

print(obj.condense(0.1))
print(obj.wordDF)
