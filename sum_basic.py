import operator
import sys

from process_data import ProcessData

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag

class SumBasic:

    def __init__(self, data):
        self.data = []

        for sentence in data:
            tokens = sent_tokenize(sentence)
            for s in tokens:
                if s not in self.data:
                    self.data.append(tokens)
        
        self.sentence_weights = {}

        data_processor = ProcessData(data)
        
        self.sentences = data_processor.remove_tags(data_processor.clean_sentences())

        self.probabilities = self._get_probabilities(self.sentences)

    def get_summary(self, length):
        
        summary = []

        while len(summary) < length and len(self.data) > 0:
            for sentence in self.sentences:
                self.sentence_weights[tuple(sentence)] = self._weight_sentence(sentence)

            winner = self._get_max_sentence()
            winner_index  = self.sentences.index(winner)

            summary.append(self.data[winner_index][0])

            if winner != '':
                self._update_probabilities(winner)
                del self.sentences[winner_index]
                del self.data[winner_index]

        return summary
    
    def _update_probabilities(self, winner):
        for word in winner:
            self.probabilities[word.lower()] = self.probabilities[word.lower()] * self.probabilities[word.lower()]

    def _get_probabilities(self, data):
        
        word_probabilities = {}
        token_count = 0

        for sentence in data:

            for word in sentence:
                token_count += 1

                if word in word_probabilities:
                    word_probabilities[word.lower()] += 1
                else:
                    word_probabilities[word.lower()] = 1

        for word in word_probabilities:
            word_probabilities[word] = word_probabilities[word] / token_count

        return word_probabilities

    def _weight_sentence(self, sentence):
        sentence_count = 0
        token_count = 0
        for word in sentence:
            if word in self.probabilities:
                sentence_count += self.probabilities[word]
                token_count += 1
        
        return sentence_count/token_count if token_count > 0 else 0

    def _get_max_sentence(self):
        highest_prob_word = max(self.probabilities.items(), key=operator.itemgetter(1))[0]

        sentences_containing_highest_prob_word = []
        for sentence in self.sentences:
            if highest_prob_word in sentence:
                sentences_containing_highest_prob_word.append(sentence)

        winner = ""
        winner_prob = 0

        for sentence in sentences_containing_highest_prob_word:
            if self.sentence_weights[tuple(sentence)] > winner_prob:
                winner_prob = self.sentence_weights[tuple(sentence)]
                winner = sentence
        
        return winner

if __name__ == "__main__":
    input_text = []
    
    with open("toy_input.txt", 'r') as toy:
        for line in toy:
            input_text.append(line)

    input_text = [s.strip() for s in input_text]

    #print(input_text)
    #sys.exit()
    summarizer = SumBasic(input_text)

    summary = summarizer.get_summary(15)

    print(summary)
    '''for sentence in summary:
        print(sentence)'''