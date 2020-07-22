import nltk
from nltk import StanfordTagger
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import pandas as pd
import string
import re
from npr_webscraper import scrape

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

    def get_sentiment(self, text):
        """
            gets the sentiment of the given text
            :return: the sentiment of the text: (positive valence - negative valence) / neutral valence
            :rtype: float
        """
        scores = self.sia.polarity_scores(text)
        return (scores['pos']-scores['neg']) / scores['neu']

    def score_word(self, word):
        # placeholder
        # should contain algorithm that weights word frequency, etc.
        # should reference self.wordDF

        stop_words = set(stopwords.words('english'))
        if word not in string.punctuation and word not in stop_words:
            return len(self.wordDF.loc[self.wordDF['Lemmas'] == self.wnl.lemmatize(word)])
        else:
            return 0

    def score_sentence(self, sentence):
        """
        Takes in a sentence and returns its score
        :param sentence: element of self.sentences
        :type sentence: str
        :return: score of sentence
        :rtype: float
        """
        # word score
        words = word_tokenize(sentence)
        score = 0
        for word in words:
            score += self.score_word(word)
        score /= len(words)

        # adding points if sentence matches overall sentiment of text

        # TODO: adjust these values to be reasonable within context of word score (is +0.5 too much/little?)
        ovr_sentiment = self.get_sentiment(self.fullText)
        if ovr_sentiment > 0.05 and self.sentencesDF.at[sentence, 'Sentiment'] > 0.4:  # positive
            score += 0.5
        elif ovr_sentiment < -0.05 and self.sentencesDF.at[sentence, 'Sentiment'] < -0.4:  # negative
            score += 0.5

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
    # I made a temporary solution to allow for raw text input; but there might be a better/more consistent way —Toby
    def __init__(self, text):
        self.wnl = WordNetLemmatizer()
        self.fullText = ""

        # regex to test if the text is a link
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        # check if text is file or not
        if text.split('.')[-1] == 'txt':
            with open(text) as f:
                for line in f:
                    self.fullText += line
        elif re.match(regex, text):
            print('scraping')
            self.fullText = scrape(text)['Raw Text']
        else:
            self.fullText = text

        self.sentences = sent_tokenize(self.fullText)


        wordData = []
        lemmas = []
        for sentence in self.sentences:
            words = word_tokenize(sentence)
            wordData.append(words)
            for word in words:
                lemmas.append(self.wnl.lemmatize(word))

        #scores words
        self.words = wordData
        self.wordDF = pd.DataFrame.from_dict({'Words': word_tokenize(self.fullText),'Lemmas': lemmas})
        self.wordDF['Scores'] = [self.score_word(word) for word in self.wordDF['Words']]

        # sentiment analysis for sentences
        self.sia = SentimentIntensityAnalyzer()
        sentence_sentiments = [self.get_sentiment(sentence) for sentence in self.sentences]
        self.sentencesDF = pd.DataFrame.from_dict({'Sentence': self.sentences,
                                                   'Sentiment': sentence_sentiments
                                                   })
        self.sentencesDF.set_index(self.sentencesDF['Sentence'], inplace=True)
        del self.sentencesDF['Sentence']




obj = WordDataFrame('https://www.npr.org/2020/07/20/891854646/whales-get-a-break-as-pandemic-creates-quieter-oceans')

print(obj.sentencesDF)
print(obj.condense(0.3))
