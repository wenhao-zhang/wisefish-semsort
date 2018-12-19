from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.tag.stanford import StanfordPOSTagger


class ProcessData:

    def __init__(self, data, type_pos_tag = "NLTK", remove_short = True, lemmatize = True, remove_stop_word = True, remove_punc = True):

        self.puncs = [".",",",";",":","!","?","``","''","'","-"]
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = stopwords.words("english")
        self.sentences = []
        self.type_pos_tag = type_pos_tag
        self.pos_tagger = StanfordPOSTagger('./stanford/models/english-bidirectional-distsim.tagger','./stanford/stanford-postagger.jar')
        self.lemmatize = lemmatize
        self.remove_stop_word = remove_stop_word
        self.remove_punc = remove_punc

        if remove_short:
            for i, line in enumerate(data):
                if len(line) < 2:
                    del data[i]

        for line in data:
            tokens = sent_tokenize(line)
            for s in tokens:
                if s not in self.sentences:
                    self.sentences.append(s)

    def clean_sentences(self):
        cleaned = []
        
        for sentence in self.sentences:
            words = word_tokenize(sentence)

            #words = [w.lower() for w in words]

            #pos tagging
            if self.type_pos_tag == "Stanford":
                tokens = []
                for w in words:
                    tokens.extend(self.pos_tagger.tag([w]))

            else:
                tokens = []
                for w in words:
                    tokens.extend(pos_tag([w]))

            tokens = [(t[0].lower(),t[1]) for t in tokens]

            #remove punctuations
            if self.remove_punc:
                tokens = self._remove_punc(tokens)

            #lemmatize
            if self.lemmatize:
                tokens = [(self.lemmatizer.lemmatize(t[0], self._get_wordnet_pos(t[1])),t[1]) for t in tokens]

            #remove stop words
            if self.remove_stop_word:
                tokens = [t for t in tokens if t[0] not in self.stopwords]

            cleaned.append(tokens)
        
        return cleaned
    
    def remove_tags(self, data):
        cleaned = []

        for s in data:
            temp = [w[0] for w in s]
            cleaned.append(temp)
        
        return cleaned

    def _get_wordnet_pos(self, treebank_tag):
        switch = {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}
        return switch.get(treebank_tag, wordnet.NOUN)

    def _remove_punc(self, sent):
        
        for punc in self.puncs:
            sent = list(filter(lambda a: a[0] != punc, sent))
       
        return sent


if __name__ == "__main__":


    data = ["The IPhone is a great device.", "My phone calls drop frequently with the IPhone.", "Great device, but the calls drop too frequently.", "The IPhone was worth the price."]
    #process = ProcessData(data)
    #t = process.clean_sentences()
    #print(t)
    #print(process.remove_tags(t))

    process_s = ProcessData(data,"Stanford", False, False, False, remove_punc = False)
    t_s = process_s.clean_sentences()
    print(t_s)
