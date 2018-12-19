from process_data import ProcessData

class OpinosisGraph():

    def __init__(self, data, remove_stop_word = False, lemmatize = True):
        self.sentences = []
        self.graph = {}
        self.PRI = {}
        processor =  ProcessData(data,"Stanford", False, lemmatize, remove_stop_word, False)

        self.sentences = processor.clean_sentences()

    def generate_opinosis_graph(self):
        n = len(self.sentences)
        for i in range(n):
            words =  self.sentences[i]
            
            for (j,(word,pos)) in enumerate(words):
                if word == "'s" and pos == "VBZ":
                    words[j] = ("is", pos)
                if word == "n't" and pos == "RB":
                    words[j] = ("not", pos)
                if word == "wa" and pos == "VBD":
                    words[j] = ("was", pos)
                if word == "ca" and pos == "MD":
                    words[j] = ("can", pos)

            sent_size = len(words)
            for j in range(sent_size):
                LABEL = words[j]
                PID = j
                SID = i
                if LABEL in self.PRI:
                    self.PRI[LABEL].append((SID, PID))
                else:
                    self.graph[LABEL] = []
                    self.PRI[LABEL] = [(SID, PID)]
                
                prev_node = words[j-1]

                if not self._edge_exists(prev_node, LABEL) and j > 0:
                    self.graph[prev_node].append(LABEL)
 
    def _edge_exists(self, prev_node, curr_node):
        if prev_node in self.graph:
            edges = self.graph[prev_node]
            if curr_node in edges:
                return True

        return False

if __name__ == "__main__":

    graph = OpinosisGraph(["The IPhone is a great device.", "My phone calls drop frequently with the IPhone.", "Great device, but the calls drop too frequently.", "The IPhone is worth the price."], False, False)
    graph.generate_opinosis_graph()
    print(graph.graph)
    print(graph.PRI)