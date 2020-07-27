from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.article import Summarizer
import random
from lexrank import STOPWORDS, LexRank
import os
import json
import gensim
from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess

class RandomSelector(Summarizer):
    def condense(self, percent):
        # automatically sets percent to condense by if not specified
        if not percent:
            percent = self.get_optimal_condense_percent()

        # calculates number of sentences to return based on input
        num_sentences = int(len(self.sentences) * percent)
        if num_sentences < 1:
            num_sentences = 1
        elif num_sentences > len(self.sentences):
            num_sentences = len(self.sentences)

        sentences = random.sample(self.sentences, len(self.sentences))
        sentences_to_return = sentences[:num_sentences]

        # sort sentences in original chronological order
        sentences_to_return.sort(key=lambda x: self.sentences.index(x))

        # joins sentences to make text body

        # list for each paragraph
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

class LexRanker(Summarizer):

    def condense(self, percent):
        # automatically sets percent to condense by if not specified
        if not percent:
            percent = self.get_optimal_condense_percent()

        # calculates number of sentences to return based on input
        num_sentences = int(len(self.sentences) * percent)
        if num_sentences < 1:
            num_sentences = 1
        elif num_sentences > len(self.sentences):
            num_sentences = len(self.sentences)

        # create corpus from docs
        dirname = os.path.dirname(__file__)
        # checks if dumped json exists; if yes, loads that
        if os.path.isfile(os.path.join(dirname, 'corpus.json')):
            with open(os.path.join(dirname, 'corpus.json'), 'r') as f:
                documents = json.load(f)
        # otherwise, creates new corpus based on files in training_data directory
        else:
            documents = make_corpus_from_files('training_data')

        lxr = LexRank(documents, stopwords=STOPWORDS['en'])

        # create summary
        sentences_to_return = lxr.get_summary(self.sentences, summary_size=num_sentences)

        # joins sentences to make text body

        # list for each paragraph
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


def make_corpus_from_files(folder_path, write=False):

    print('Creating corpus...')
    documents = []
    dirname = os.path.dirname(__file__)
    for file_name in os.listdir(os.path.join(dirname, folder_path)):
        file_path = os.path.join(dirname, folder_path, file_name)
        with open(file_path, mode='rt', encoding='utf-8') as f:
            documents.append(f.readlines())

    if write:
        with open(os.path.join(dirname, 'corpus.json'), mode='w') as f:
            json.dump(documents, f, indent=4)
    return documents


def score_summarizer(summarizer, percent, vectorizer=None):
    documents = [summarizer.fullText, summarizer.condense(percent)]

    if not vectorizer:
        vectorizer = CountVectorizer(stop_words='english')
    sparse_matrix = vectorizer.fit_transform(documents)

    return cosine_similarity(sparse_matrix)[0][1]


def soft_score(summarizer, percent):
    documents = [summarizer.fullText, summarizer.condense(percent)]

    print('loading')
    fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

    dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])
    print(dictionary)
    # similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0,
    #                                                         exponent=2.0, nonzero_limit=100)
    #
    # original_doc = dictionary.doc2bow(simple_preprocess(documents[0]))
    # condensed_doc = dictionary.doc2bow(simple_preprocess(documents[1]))
    #
    # return softcossim(original_doc, condensed_doc, similarity_matrix)

def find_optimal_params(summarizer):
    results = []
    print()
    vector = CountVectorizer(stop_words='english')
    for word_freq in range(0, 5, 1):
        for vec in range(0, 5, 1):
            for sentiment in range(0, 5, 1):
                for similarity in range(0, 5, 1):
                    out = f'\rWord frequency: {word_freq} Vector: {vec} Sentiment: {sentiment} Similarity: {similarity}'
                    print(out, end='')
                    summarizer.set_weights({'Word Frequency': word_freq,
                                            'Vector': vec,
                                            'Sentiment': sentiment,
                                            'Similarity': similarity})
                    score = score_summarizer(summarizer, summarizer.get_optimal_condense_percent(), vectorizer=vector)
                    results.append(((word_freq, vec, sentiment, similarity), score))
    print()
    return results

if __name__ == '__main__':
    # make_corpus_from_files('training_data', write=True)
    url = 'https://www.npr.org/2020/07/20/891854646/whales-get-a-break-as-pandemic-creates-quieter-oceans'
    summarizer = Summarizer(url)
    param_results = find_optimal_params(summarizer)
    param_results.sort(key=lambda x: x[1])
    print(param_results)