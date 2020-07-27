from nltk.translate.bleu_score import corpus_bleu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.article import Summarizer
import random
from lexrank import STOPWORDS, LexRank
import os
import json

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


def score(summarizer, percent):
    documents = [summarizer.fullText, summarizer.condense(percent)]

    count_vectorizer = CountVectorizer(stop_words='english')
    sparse_matrix = count_vectorizer.fit_transform(documents)

    return cosine_similarity(sparse_matrix)[0][1]


if __name__ == '__main__':
    # make_corpus_from_files('training_data', write=True)
    url = 'https://www.npr.org/2020/07/20/891854646/whales-get-a-break-as-pandemic-creates-quieter-oceans'
    summarizer = Summarizer(url)
    print(f'Our summarizer: {score(summarizer, 0.2)}')
    randomizer = RandomSelector(url)
    print(f'Random sentence selection: {score(randomizer, 0.2)}')
    lexranker = LexRanker(url)
    print(f'LexRank selection: {score(lexranker, 0.2)}')