import nltk
from nltk import StanfordTagger
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.wsd import lesk
import pandas as pd
import string
import re
from npr_webscraper import scrape
from fuzzywuzzy import fuzz
from gensim.models import Word2Vec
from nltk.corpus import brown


class WordDataFrame:


    def score_word(self, word):
        """
        Takes in a word and returns its score
        :param word: element of self.words
        :type sentence: str
        :return: score of word
        :rtype: float
        """
        stop_words = set(stopwords.words('english'))
        if word not in string.punctuation and word not in stop_words:
            return len(self.wordDF.loc[self.wordDF['Lemmas'] == self.wnl.lemmatize(word)])
        else:
            return 0
    def get_tags(self, row):
        '''
            tags words by part of speech
            inputs: self, row number (int)
            outputs:  words and their part of speech (array[tuples])
        '''
        tagged = nltk.pos_tag(nltk.word_tokenize(self.sentences[row]))
        return tagged
    def get_verbs(self, row):
        '''
            finds verbs in a sentence
            inputs: self, row number (int)
            outputs:  words and their part of speech (array[VERBS])
        '''
        verbs = []
        tagged = self.getTagsByRow(row)
        for word in tagged:
            if word[1] == "VB":
                verbs.append(word[0])
        return verbs
    def get_nouns(self, row):
        '''
            finds nouns in a sentence
            inputs: self, row number (int)
            outputs:  words and their part of speech (array[Nouns])
        '''
        nouns = []
        tagged = self.getTagsByRow(row)
        for word in tagged:
            if word[1] == "NN":
                nouns.append(word[0])
        return nouns

    def get_similarity(self, sentence, currentScore):
        """
        Takes in a sentence and returns how similar it is to another sentence
        :param sentence: element of self.sentences
        :type sentence: str
        :return: score of sentence
        :rtype: float
        """
        index = self.sentences.index(sentence)

        if index != len(self.sentences)-1:
            scores = [fuzz.token_sort_ratio(sim, sentence) for sim in self.sentences[index+1:]]
            adjustedScore = -max(scores)/100
            if adjustedScore < -0.85: return 2 * adjustedScore #heavily decrecrement highly similar sentences
            elif adjustedScore < -0.6: return adjustedScore #decrement the score of somewhat similar sentences
            else: return 0 #ignore low levels of similarity
        else:
            return 0

    def get_sentiment(self, text):
        """
            gets the sentiment of the given text
            :return: the sentiment of the text: (positive valence - negative valence) / neutral valence
            :rtype: float
        """
        scores = self.sia.polarity_scores(text)
        if scores['neu'] < 0.1:
            scores['neu'] = 0.1
        return (scores['pos'] - scores['neg']) / scores['neu']

    def score_synset(self, synset):
        """
            Takes in a synset and returns its score
            :param synset: The synset of a word
            :type synset: wn.synset
            :return: The score of the given synset (# of times it appears in the text)
            :rtype: int
        """
        if synset is None: # Nones should be removed, leaving it just in case
            return 0
        else:
            return len(self.wordDF[self.wordDF['Synsets'] == synset])

    def score_word2vec(self,word):
        score = 0
        for sim in self.vec.wv.most_similar(word, topn=3):
            score += sim[1]
        return score

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
            word_synset = lesk(words, word)
            score += self.score_synset(word_synset)
            if word in self.word_sentences: score += self.score_word2vec(word)
        score /= len(words)

        # adding points if sentence matches overall sentiment of text

        # TODO: adjust these values to be reasonable within context of word score (is +0.5 too much/little?)
        ovr_sentiment = self.get_sentiment(self.fullText)
        if ovr_sentiment > 0.05 and self.sentencesDF.at[sentence, 'Sentiment'] > 0.4:  # positive
            score += 0.5
        elif ovr_sentiment < -0.05 and self.sentencesDF.at[sentence, 'Sentiment'] < -0.4:  # negative
            score += 0.5

        #subtracting points based on high similarity to other sentences
        score += self.get_similarity(sentence,score)
        print(score)
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

    def __init__(self, text):
        """
        Constructor
        :param text: text input
        :type percent: file or str
        """
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

        # check if text is file or link or raw
        if text.split('.')[-1] == 'txt':
            with open(text) as f:
                for line in f:
                    self.fullText += line
        elif re.match(regex, text):
            self.fullText = scrape(text)['Raw Text']
        else:
            self.fullText = text

        self.sentences = sent_tokenize(self.fullText)


        # adds words and their lemmas and synsets to these lists
        wordData = []
        lemmas = []
        synsets = []
        self.word_sentences = []

        # also filters stopwords/punctuation
        stop_words = set(stopwords.words('english'))

        for sentence in self.sentences:
            words = word_tokenize(sentence)
            for word in words:
                if word not in string.punctuation and word not in stop_words:
                    wordData.append(word)
                    lemmas.append(self.wnl.lemmatize(word))
                    synsets.append(lesk(words, word))
                    self.word_sentences.append(word)

        self.wordDF = pd.DataFrame.from_dict({'Words': wordData,
                                              'Lemmas': lemmas,
                                              'Synsets': synsets})

        # removes "None"s from df
        self.wordDF = self.wordDF[self.wordDF['Synsets'].notnull()]

        # sentiment analysis for sentences
        self.sia = SentimentIntensityAnalyzer()
        sentence_sentiments = [self.get_sentiment(sentence) for sentence in self.sentences]
        self.sentencesDF = pd.DataFrame.from_dict({'Sentence': self.sentences,
                                                   'Sentiment': sentence_sentiments
                                                   })
        self.sentencesDF.set_index(self.sentencesDF['Sentence'], inplace=True)
        del self.sentencesDF['Sentence']

        self.vec = Word2Vec([self.word_sentences], min_count=1)





#obj = WordDataFrame('https://www.npr.org/2020/07/20/891854646/whales-get-a-break-as-pandemic-creates-quieter-oceans')
obj = WordDataFrame('test.txt')

print(obj.condense(0.3))
#obj.score_word2vec()
