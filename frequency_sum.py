from nltk.tokenize import sent_tokenize, word_tokenize
import operator

def run(input_text):
    for i, line in enumerate(input_text):
        if len(line) < 2:
            del input_text[i]

    word_prob = {}
    words = []
    sentences = []

    # step 1 - needs punctuation pruning
    for line in input_text:
        token_sentences = sent_tokenize(line)
        sentences = sentences + token_sentences

    for sent in sentences:
        sent_words = word_tokenize(sent)
        words = words + sent_words

    for word in words:
        if word in word_prob:
            word_prob[word.lower()] += 1
        else:
            word_prob[word.lower()] = 1

    for word in word_prob:
        word_prob[word] = word_prob[word] / len(words)

    # step 2

    sentence_weights = {}

    def weight_sentences(sentence_weights, sentences):
        for sentence in sentences:
            sentence_prob = 0.0
            for word in word_tokenize(sent):
                sentence_prob += word_prob[word.lower()]
            sentence_weights[sentence] = sentence_prob / len(sentence)

    # step 3
    def find_highest_scoring_sentence(sentences, word_prob, sentence_weights,):
        highest_prob_word = max(word_prob.items(), key=operator.itemgetter(1))[0]

        sentences_containing_highest_prob_word = []
        for sentence in sentences:
            if highest_prob_word in sentence:
                sentences_containing_highest_prob_word.append(sentence)

        winner = ""
        winner_prob = 0

        for sentence in sentences_containing_highest_prob_word:
            if sentence_weights[sentence] > winner_prob:
                winner_prob = sentence_weights[sentence]
                winner = sentence
        return winner

    summary = []
    while len(summary) < 3 and len(sentences) > 0:
        weight_sentences(sentence_weights, sentences)
        winner = find_highest_scoring_sentence(sentences, word_prob, sentence_weights)
        summary.append(winner)
        if winner != '':
            del sentences[sentences.index(winner)]
            for word in word_tokenize(winner):
                word_prob[word.lower()] = word_prob[word.lower()] * word_prob[word.lower()]

    return summary

if __name__ == "__main__":
    input_text = []

    with open("toy_input.txt", 'r') as toy:
        for line in toy:
            input_text.append(line)

    summary = run(input_text)

    for sentence in summary:
        print(sentence)
