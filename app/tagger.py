from sklearn.feature_extraction.text import TfidfVectorizer
from webscraper import scrape
import os
import json
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def create_json_corpus(filepath='corpus.json'):
    """
    Creates a list of sentences from the BBC dataset and saves it as a JSON file. The main BBC folder should be
    located in the article-summarizer folder.

    :param filepath: The location at which to save the JSON file.
    :type filepath: str
    :return: The list of all sentences from the dataset.
    :rtype: list
    """
    bbc_corpora = []
    project_folder = os.path.dirname(os.path.dirname(__file__))
    for folder in ['bbc/business', 'bbc/entertainment', 'bbc/politics', 'bbc/sport', 'bbc/tech']:
        data_folder = os.path.join(project_folder, folder)
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                with open(os.path.join(data_folder, file), 'r') as f:
                    bbc_corpora.append(f.readlines())

    bbc_corpora = [sentence for article in bbc_corpora for sentence in article]

    # maybe add files from database here?
    corpus = bbc_corpora

    with open(filepath, 'w') as f:
        json.dump(corpus, f, indent=4)

    return corpus


def load_json_corpus(filepath='corpus.json'):
    """
    Loads an existing corpus from a JSON file.

    :param filepath: The location of the file to load.
    :type filepath: str
    :return: The list of all sentences from the dataset.
    :rtype: list
    """
    with open(filepath) as f:
        return json.load(f)


def get_tags(text, n):
    """
    Finds n keywords for a given piece of text.

    :param text: The text to extract keywords from.
    :type text: str
    :param n: The number of keywords to return.
    :type n: int
    :return: List of n keywords sorted in order of importance.
    :rtype: list
    """

    # load corpora
    try:
        corpus = load_json_corpus()
    except FileNotFoundError:
        corpus = create_json_corpus()

    # extract nouns from text

    tokenized_text = word_tokenize(text)
    nouns = [word for word, pos in pos_tag(tokenized_text) if pos[:2] == 'NN']
    noun_string = ' '.join(nouns)

    corpus = [noun_string] + corpus

    vectorizer = TfidfVectorizer(use_idf=True)
    matrix = vectorizer.fit_transform(corpus)

    first_vector = matrix[0]

    first_vector_scores = first_vector.T.todense().tolist()

    results = [(name, score[0]) for name, score in zip(vectorizer.get_feature_names(), first_vector_scores)]
    results.sort(reverse=True, key=lambda x: x[1])

    return [result[0] for result in results[:n] if result[1] > 0]


if __name__ == '__main__':
    url = 'https://www.cnn.com/2020/07/29/us/football-player-doctor-covid-concerns/index.html'
    text = scrape(url)['Title']

    print(get_tags(text, 5))
