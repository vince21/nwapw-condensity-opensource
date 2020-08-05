# NWAPW
# Spencer Chang, Toby Ueno, Vincent Wilson
# date: 8/04/20
# description: this is the file of the Summarizer class

from nltk import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.wsd import lesk
import pandas as pd
import string
from urllib.parse import urlparse
from webscraper import scrape
from webscraper import is_valid_url
from fuzzywuzzy import fuzz
from gensim.models import Word2Vec
from datetime import datetime
import numpy as np


class Summarizer:
    """
    This class summarizes text inputs
    Attributes:
        fullText (str): the entire text input

        title (str): attribute if scraping article
        authors (str): attribute if scraping article
        date (str): attribute if scraping article
        image (str): attribute if scraping article
        domain (str): attribute if scraping article

        sentences ([str]): sentence tokenized full text
        paragraphs ([str]): paragraph tokenized full text

        all_words ([str]): contains all words including stop words
        wordDF (DataFrame): contains words and their corresponding synsets
        wordlist ([str]): list of all words - stopwords and punctuation

        sia (SentimentIntensityAnalyzer): sentiment analysis object
        sentencesDF (DataFrame): sentences and their sentiment

        vec (Word2Vec): word2vec trained object
        weights (dict): default weightages
    """

    def sanitize_text(self):
        """
        Takes in full text and handles punctuation like parentheses
        """
        sentences = sent_tokenize(self.fullText)
        good_sentences = []
        for sentence in sentences:
            valid_sentence = True
            paren_variants = ['(', '[', '{', '-', '—']
            words = word_tokenize(sentence)
            if words[:3] == [word.upper() for word in words[:3]]:
                valid_sentence = False
            elif sentence[0] in paren_variants:
                valid_sentence = False
            if valid_sentence:
                good_sentences.append(sentence)

        self.fullText = '\n'.join(good_sentences)

    def get_similarity(self, sentence):
        """
        Takes in a sentence and returns how similar it is to another sentence
        :param sentence: element of self.sentences
        :type sentence: str
        :return: score of sentence
        :rtype: float
        """
        index = self.sentences.index(sentence)

        if index != len(self.sentences) - 1:
            scores = [fuzz.token_sort_ratio(sim, sentence) for sim in self.sentences[index + 1:]]
            adjusted_score = -max(scores) / 100
            if adjusted_score < -0.85:
                return 2 * adjusted_score  # heavily decrement highly similar sentences
            elif adjusted_score < -0.6:
                return adjusted_score  # decrement the score of somewhat similar sentences
            else:
                return 0  # ignore low levels of similarity
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
        if synset is None:  # Nones should be removed, leaving it just in case
            return 0
        else:
            return len(self.wordDF[self.wordDF['Synsets'] == synset])

    def score_word2vec(self, word):
        """
        Takes in a word and adds up it's top 3 most similar word2vecs
        :param word: a word in the vocab
        :type sentence: str
        :return: word2vec score of word
        :rtype: float
        """
        score = 0
        if word not in self.vec.wv:
            return 0
        for sim in self.vec.wv.most_similar(word, topn=3):
            if sim[1] > 0.3:
                score += sim[1]
        return score

    def set_weights(self, new_weights):
        for key, value in new_weights.items():
            self.weights[key] = value

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
            score += self.score_synset(word_synset) * self.weights['Word Frequency']
            # if the word is in the vocab
            if word in self.wordlist:
                score += self.score_word2vec(word) * self.weights['Vector']
        score /= len(words)

        # adding points if sentence matches overall sentiment of text

        ovr_sentiment = self.get_sentiment(self.fullText)
        if ovr_sentiment > 0.05 and self.sentencesDF.at[sentence, 'Sentiment'] > 0.4:  # positive
            score += 1 * self.weights['Sentiment']
        elif ovr_sentiment < -0.05 and self.sentencesDF.at[sentence, 'Sentiment'] < -0.4:  # negative
            score += 1 * self.weights['Sentiment']

        # subtracting points based on high similarity to other sentences
        score += self.get_similarity(sentence) * self.weights['Similarity']

        # subtracting points for short sentences (with the goal of removing subtitles/similar)
        if len(words) <= 1:
            score /= 20
        elif len(words) == 2:
            score /= 10

        return score

    def condense(self, percent=None):
        """
        Condenses the text in self.sentences by a given percent
        :param percent: Specifies how much the text should be reduced by (0 < percent <= 1)
        :type percent: float
        :return: String containing abbreviated text
        :rtype: str
        """
        # automatically sets percent to condense by if not specified
        if not percent:
            percent = self.get_optimal_condense_percent()

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

        # create list for each paragraph
        output = [[] for _ in self.paragraphs]

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

    def get_optimal_condense_percent(self):
        """
        Calculates a percent to condense to based on the number of sentences
        :return: percent to condense to
        :rtype: float
        """
        target_sentences = np.log(len(self.sentences)) ** 2
        if target_sentences > 25:  # happens at around 150 sentences
            target_sentences = 25
        return target_sentences / len(self.sentences)

    def condense_metrics(self, condensed_text):
        """
        Gets info on how much the text was condensed by.
        :param condensed_text: The condensed text (to be compared against the original)
        :type condensed_text: str
        :return: The absolute and relative reductions of sentences, words, and characters.
        :rtype: dict
        """
        og_info = {'Sentences': len(self.sentences),
                   'Words': len(self.all_words),
                   'Characters': len(self.fullText)}
        condensed_info = {'Sentences': len(sent_tokenize(condensed_text)),
                          'Words': len(word_tokenize(condensed_text)),
                          'Characters': len(condensed_text)}

        info = {k: og_info[k] - condensed_info[k] for k in og_info.keys()}
        percentage_info = {'% ' + k: 100 * (1 - round(condensed_info[k] / og_info[k], 4)) for k in og_info.keys()}
        info.update(percentage_info)
        info['Total %'] = round(sum(percentage_info.values()) / len(percentage_info.values()), 2)
        info = {k: max(v, 0) for k, v in info.items()}
        return info

    def get_article_info(self):
        """
        Gets non-body info about the article. If a non-URL source is used, will return None for all keys except "Text".
        :return: Dict containing "Title", "Authors" (list), "Date" (datetime), "Text", "Image"
        :rtype: dict
        """
        return {'Title': self.title,
                'Authors': self.authors,
                'Date': self.date,
                'Text': self.fullText,
                'Image': self.image}

    def __init__(self, text, weights=None):
        """
        Constructor
        :param text: text input
        :type text: file or str
        """
        self.fullText = ""

        # attributes that can be scraped (if a link is inputted)
        self.title = None
        self.authors = None
        self.date = None
        self.image = None
        self.domain = None

        # check if text is file or link or raw
        if text.split('.')[-1] == 'txt':
            with open(text) as f:
                for line in f:
                    self.fullText += line
        elif is_valid_url(text):
            scrape_results = scrape(text)
            self.fullText = scrape_results['Text']
            self.sanitize_text()
            self.title = scrape_results['Title']
            self.authors = scrape_results['Authors']
            self.image = scrape_results['Image']
            self.domain = urlparse(text).netloc
        else:
            self.fullText = text

        self.sentences = sent_tokenize(self.fullText)

        self.paragraphs = [sent_tokenize(paragraph) for paragraph in self.fullText.split('\n') if paragraph != '']
        self.sentences = [sentence for paragraph in self.paragraphs for sentence in paragraph]

        # adds words and their lemmas and synsets to these lists
        word_data = []
        lemmas = []
        synsets = []

        # used for finding condense percent; not for use in scoring (contains stopwords, etc.)
        self.all_words = []

        # also filters stopwords/punctuation
        stop_words = set(stopwords.words('english'))

        for sentence in self.sentences:
            words = word_tokenize(sentence)
            for word in words:
                self.all_words.append(word)
                if word not in string.punctuation and word not in stop_words:
                    word_data.append(word)
                    synsets.append(lesk(words, word))

        #words and corresponding synsets
        self.wordDF = pd.DataFrame.from_dict({'Words': word_data,
                                              'Synsets': synsets})

        self.wordlist = word_data

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

        self.vec = Word2Vec([self.wordlist], min_count=1)

        # set default scoring weights
        self.weights = {'Word Frequency': 1,
                        'Vector': 4,
                        'Sentiment': 1,
                        'Similarity': 1}


if __name__ == '__main__':
    start_time = datetime.now()
    obj = Summarizer('https://www.npr.org/2020/07/20/891854646/whales-get-a-break-as-pandemic-creates-quieter-oceans')
    condensed_text = obj.condense()
    print(condensed_text)
    print(obj.condense_metrics(condensed_text))
    print(f'\nTime: {datetime.now() - start_time}')
