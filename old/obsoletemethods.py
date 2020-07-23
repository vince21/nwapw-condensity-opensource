from nltk import StanfordTagger
from nltk.stem import WordNetLemmatizer

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
