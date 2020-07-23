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
from fuzzywuzzy import fuzz

class WordDataFrame:


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

    def get_similarity(self, sentence):
        """
        Takes in a sentence and returns how similar it is to another sentence
        :param sentence: element of self.sentences
        :type sentence: str
        :return: score of sentence
        :rtype: float
        """
        scores = []
        for sim in self.sentences[self.sentences.index(sentence)+1:]:
            scores.append(fuzz.token_sort_ratio(sim, sentence))
        #TODO: fix this because it breaks
        try:
            return -max(scores)/100
        except ValueError:
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

        score += self.get_similarity(sentence)

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

        # list for each paragraph
        output = [[] for i in self.paragraphs]

        # copies self.paragraphs to prevent destructive edits
        paragraphs = [paragraph[:] for paragraph in self.paragraphs]

        for sentence in sentences_to_return:
            for i, paragraph in enumerate(paragraphs):
                if sentence in paragraph:
                    output[i].append(sentence)
                    paragraph.remove(sentence)
                    break

        # joins paragraph sentences with spaces
        output = [' '.join(paragraph) for paragraph in output]
        # joins paragraphs with newlines if paragraphs aren't empty
        output = '\n\n'.join([x for x in output if x.strip() != ''])

        return output

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

        self.paragraphs = [sent_tokenize(paragraph) for paragraph in self.fullText.split('\n') if paragraph != '']
        self.sentences = [sentence for paragraph in self.paragraphs for sentence in paragraph]


        # adds words and their lemmas to these lists
        wordData = []
        lemmas = []
        for sentence in self.sentences:
            words = word_tokenize(sentence)
            for word in words:
                wordData.append(word)
                lemmas.append(self.wnl.lemmatize(word))

        #scores words
        self.words = wordData
        self.wordDF = pd.DataFrame.from_dict({'Words': self.words,'Lemmas': lemmas})
        self.wordDF['Scores'] = [self.score_word(word) for word in self.wordDF['Words']]

        # sentiment analysis for sentences
        self.sia = SentimentIntensityAnalyzer()
        sentence_sentiments = [self.get_sentiment(sentence) for sentence in self.sentences]
        self.sentencesDF = pd.DataFrame.from_dict({'Sentence': self.sentences,
                                                   'Sentiment': sentence_sentiments
                                                   })
        self.sentencesDF.set_index(self.sentencesDF['Sentence'], inplace=True)
        del self.sentencesDF['Sentence']





obj = WordDataFrame('Test line. This is a test. Test line.\nTest line. This is an experiment.')

#obj = WordDataFrame('test.txt')

print(obj.condense(0.75))
