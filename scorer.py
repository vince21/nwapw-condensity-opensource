from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.article import Summarizer
import random
from lexrank import STOPWORDS, LexRank
import os
import json
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from gensim.similarities.docsim import SoftCosineSimilarity, SparseTermSimilarityMatrix
from gensim.models.keyedvectors import WordEmbeddingSimilarityIndex
from nltk import word_tokenize
from datetime import datetime


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


def create_model():
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, 'corpus.json'), 'r') as f:
        articles = json.load(f)
    # flatten
    sentences = [item for article in articles for item in article]
    sentences = [word_tokenize(sentence) for sentence in sentences]
    model = Word2Vec(sentences)
    return model


def soft_score_summarizer(summarizer, percent):
    documents = [summarizer.fullText, summarizer.condense(percent)]

    dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])

    model = create_model()
    termsim_index = WordEmbeddingSimilarityIndex(model.wv)

    original_doc = dictionary.doc2bow(simple_preprocess(documents[0]))
    condensed_doc = dictionary.doc2bow(simple_preprocess(documents[1]))

    similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)
    docsim_index = SoftCosineSimilarity([original_doc], similarity_matrix)

    return docsim_index[condensed_doc][0]


def find_optimal_params(summarizer):
    results = []
    print()
    TRIALS_PER_WEIGHT_SET = 2
    starting_time = datetime.now()
    vector = CountVectorizer(stop_words='english')
    for word_freq in range(0, 45, 5):
        word_freq /= 10.
        for vec in range(0, 45, 5):
            vec /= 10.
            for sentiment in range(0, 45, 5):
                sentiment /= 10.
                for similarity in range(0, 45, 5):
                    similarity /= 10.
                    out = f'\rWord frequency: {word_freq} Vector: {vec} Sentiment: {sentiment} Similarity: {similarity}'
                    out += f' Time elapsed: {datetime.now() - starting_time}'
                    print(out, end='')
                    summarizer.set_weights({'Word Frequency': word_freq,
                                            'Vector': vec,
                                            'Sentiment': sentiment,
                                            'Similarity': similarity})

                    score = 0
                    for i in range(TRIALS_PER_WEIGHT_SET):
                        score += score_summarizer(summarizer, summarizer.get_optimal_condense_percent(),
                                                  vectorizer=vector)
                    score /= TRIALS_PER_WEIGHT_SET
                    results.append(((word_freq, vec, sentiment, similarity), score))
    print()
    return results


def compare_scores(soft=False):
    url = 'https://www.npr.org/2020/07/20/891854646/whales-get-a-break-as-pandemic-creates-quieter-oceans'
    summarizer = Summarizer(url)
    randomizer = RandomSelector(url)
    lexranker = LexRanker(url)
    if soft:
        print(f'Our summarizer: {soft_score_summarizer(summarizer, 0.2)}')
        print(f'Random sentences: {soft_score_summarizer(randomizer, 0.2)}')
        print(f'LexRank: {soft_score_summarizer(lexranker, 0.2)}')
    else:
        print(f'Our summarizer: {score_summarizer(summarizer, 0.2)}')
        print(f'Random sentences: {score_summarizer(randomizer, 0.2)}')
        print(f'LexRank: {score_summarizer(lexranker, 0.2)}')

if __name__ == '__main__':
    # make_corpus_from_files('training_data', write=True)
    url = 'https://www.npr.org/2020/07/20/891854646/whales-get-a-break-as-pandemic-creates-quieter-oceans'
    summarizer = Summarizer(url)
    param_results = find_optimal_params(summarizer)
    param_results.sort(key=lambda x: x[1], reverse=True)
    print(param_results)
