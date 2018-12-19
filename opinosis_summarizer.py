import re
import math
import sys

from opinosis_graph import OpinosisGraph
from nltk import word_tokenize, pos_tag


class OpinosisSummarizer():

    def __init__(self, data, sigma_ss, sigma_r, sigma_gap, sigma_vsn, collapse, similarity, remove_stop_word=False, lemmatize=True):
        self.sigma_ss = sigma_ss
        self.sigma_r = sigma_r
        self.sigma_gap = sigma_gap
        self.sigma_vsn = sigma_vsn
        self.collapse = collapse
        self.similarity = similarity

        graph = OpinosisGraph(data, remove_stop_word, lemmatize)
        graph.generate_opinosis_graph()

        self.graph = graph.graph
        self.PRI = graph.PRI

    def _is_valid_start_node(self, node):
        PRI_node = self.PRI[node]

        common_start_words = "r^(its/|the/|when/|a/|an/|this/|the/|they/|it/|i/|we/|our/).*"

        valid_start_tags = ["/JJ", "/RB", "/PRP$", "/VBG", "/NN", "/DT"]

        len_PRI = len(PRI_node)

        total = 0

        for PRI in PRI_node:
            total += PRI[1]

        word_tag = node[0]+"/"+node[1]

        if total/len_PRI <= self.sigma_vsn:
            if re.match(common_start_words, word_tag, re.I) or "it/PRP" in word_tag or "if/" in word_tag or "for/" in word_tag:
                return True
            for v_t in valid_start_tags:
                if v_t in word_tag:
                    return True

    def _is_valid_end_node(self, node):
        if node[1] == "." or node[1] == "," or node[1] == "CC":
            return True

        return False

    def _is_valid_path(self, sentence):

        well_formed = False

        pos_sent = ""

        for w in sentence:
            pos_sent += "/" + w[1]

        if re.match(".*(/JJ)*.*(/NN)+.*(/VB)+.*(/JJ)+.*", pos_sent, re.I):
            well_formed = True
        elif not re.match(".*(/DT).*", pos_sent, re.I) and re.match(".*(/RB)*.*(/JJ)+.*(/NN)+.*", pos_sent, re.I):
            well_formed = True
        elif re.match(".*(/PRP|/DT)+.*(/VB)+.*(/RB|/JJ)+.*(/NN)+.*", pos_sent, re.I):
            well_formed = True
        elif re.match(".*(/JJ)+.*(/TO)+.*(/TO)+.*(/VB).*", pos_sent, re.I):
            well_formed = True
        elif re.match(".*(/RB)+.*(/IN)+.*(/NN)+.*", pos_sent, re.I):
            well_formed = True

        last = sentence[-1][1]

        #if re.match("(/TO|/VBZ|/IN|/CC|/PRP|/DT|/,)", last, re.I):
        #    well_formed = False

        return well_formed

    def _Swt_loglen(self, redudancy, L):
        return math.log(L, 2)*redudancy if L > 1 else redudancy

    def _is_collapsible(self, node):
        return bool(re.match(r"VB", node[1], re.I))

    def _intersect(self, PRI_n, PRI_overlap):
        PRI_intersect = []
        for sid_o, pid_o in PRI_overlap:
            for sid, pid in PRI_n:
                if sid_o == sid:
                    if pid > pid_o and abs(pid - pid_o) <= self.sigma_gap:
                        PRI_intersect.extend([(sid, pid)])
                        break
                else:
                    if sid > sid_o:
                        break

        return PRI_intersect

    def _jaccard_similarity(self, candidate1, candidate2):

        if set(candidate1).issubset(set(candidate2)) or set(candidate2).issubset(set(candidate1)):
            return 1

        intersection_size = len(set(candidate1).intersection(set(candidate2)))
        union_size = len(set(candidate1).union(set(candidate2)))
        return intersection_size/union_size

    def _remove_duplicates(self, candidates):

        clusters = [[key] for key, value in candidates.items()]
        final_sentences = []

        prev_size = len(clusters)
        curr_size = prev_size * 2
        while curr_size != prev_size and len(clusters) > 1:

            prev_size = len(clusters)

            scores = []
            temp = []

            size_cluster = len(clusters)

            for i in range(size_cluster):

                c1 = clusters[i]

                for j in range(i + 1, size_cluster):

                    c2 = clusters[j]
                    best_score = self._jaccard_similarity(c1[0], c2[0])

                    for a in c1:
                        for b in c2:
                            score = self._jaccard_similarity(a, b)
                            if score > best_score:
                                best_score = score

                    key = c1+c2
                    temp.append([c1, c2])
                    scores.append((key, best_score))

            highest = max(scores, key=lambda x: x[1])
            highest_index = scores.index(highest)

            if highest[1] > self.similarity:
                clusters.append(highest[0])

                for i in temp[highest_index]:
                    clusters.remove(i)

            curr_size = len(clusters)

        for cluster in clusters:
            max_sentence = max(cluster, key=candidates.get)
            final_sentences.append(tuple(max_sentence))

        return final_sentences

    def _average_path_score(self, cc, score):

        len_cc = len(cc)

        scores = 0

        for i in cc:
            scores += score[i]

        return scores/len_cc

    def _stitch(self, anchor, cc):
        new_sent = anchor[:]

        if len(cc) > 1:
            for s in cc[:-1]:
                for w in s:
                    new_sent += (w,)
                new_sent += ((",", ","),)
            last_node = cc[-1][0]

            cc_nodes = [("and", "CC"), ("for", "IN"), ('nor', 'CC'),
                        ('but', 'CC'), ('or', 'CC'), ('yet', 'RB'), ('so', 'RB')]

            highest_r = 0
            best_cc = cc_nodes[0]

            for node in cc_nodes:
                try:
                    r = len(self._intersect(
                        self.PRI[node], self.PRI[last_node]))
                    if r > highest_r:
                        highest_r = r
                        best_cc = node
                except(KeyError):
                    pass

            new_sent += (best_cc,)

            for w in cc[-1]:
                new_sent += (w,)

        else:
            for w in cc[0]:
                new_sent += (w,)

        return new_sent

    def _traverse(self, candidates, node, score, PRI_overlap, sentence, length):
        redundancy = len(PRI_overlap)
        if redundancy >= self.sigma_r:
            if self._is_valid_end_node(node):
                if self._is_valid_path(sentence):
                    final_score = score/length
                    candidates[sentence] = final_score

            for v_n in self.graph[node]:
                PRI_new = self._intersect(self.PRI[v_n], PRI_overlap)
                r = len(PRI_new)
                new_sentence = sentence + (v_n,)
                L = length + 1
                new_score = score + self._Swt_loglen(r, L)
                if self._is_collapsible(v_n) and self.collapse:
                    c_anchor = new_sentence
                    tmp = {}
                    for v_x in self.graph[v_n]:
                        self._traverse(tmp, v_x, 0, PRI_new, (v_x,), L)

                        if tmp:
                            cc = self._remove_duplicates(tmp)
                            cc_path_score = self._average_path_score(cc, tmp)
                            final_score = new_score + cc_path_score
                            sitch_sent = self._stitch(c_anchor, cc)
                            print(sitch_sent)
                            candidates[sitch_sent] = final_score
                else:
                    self._traverse(candidates, v_n, new_score,
                                   PRI_new, new_sentence, L)

    def get_summary(self):
        candidates = {}
        summaries = []

        for v_j in self.graph.keys():

            if self._is_valid_start_node(v_j):

                path_length = 1
                score = 0
                clist = {}
                self._traverse(clist, v_j, score,
                               self.PRI[v_j], (v_j,), path_length)
                candidates.update(clist)

        final = self._remove_duplicates(candidates)
        top = sorted(final, key = candidates.get, reverse=True)[:self.sigma_ss]
        #top = sorted(final, key=candidates.get, reverse=True)

        for s in top:
            summary = ""
            length = len(s)
            for i in range(length-1):
                summary += s[i][0]

                if s[i+1][1] == ".":
                    summary += s[i+1][0]
                elif (s[i+1][0] == "," or s[i+1][0] == "and") and i + 1 == length:
                    summary += "."
                elif s[i+1][1] != ",":
                    summary += " "

            summaries.append(summary)

        return summaries


if __name__ == "__main__":
    #data = ['The IPhone is a great device.',"My phone calls drop frequently with the IPhone.","Great device, but the calls drop too frequently.","The iPHone is worth the price."]

    #data = [' The room was what we were expecting .', 'The room was small, but clean .', 'Rooms in the hotel are very small, as is the double bed .', ' The bathroom is a good size .', ' The room was large for a London hotel with a great bathroom ,  nice tub and great shower .']

    #data = [', and is very, very accurate .', 'Accuracy is determined by the maps .', 'This is a great GPS, it is so easy to use and it is always accurate .', 'The directions are highly accurate down to a  T  .', 'Most of the times, this info was very accurate .', "0 out of 5 stars GPS Navigator doesn't navigate accurately on a straight road .", "I've even used it in the  pedestrian  mode, and it's amazing how accurate it is .", "Accuracy is as good as any other unit, they all sometimes tell you you have arrived when you haven't, or continue to tell you to turn when you're already there .", 'It got me from point A to point B with 100% accuracy everytime .',
    #        'It has worked well for local driving giving accurate directions for roads and streets .', "I can't believe how accurate and detailed the information estimated time of arrival,speed limits along the way,and detailed map of my route, to name a few .", "I found the maps to be inaccurate at first, but after I updated them from Garmin's website everything is golden .", 'The closest one that gives the most accurate route that I usually take is the Navigon .', 'DESTINATION TIME, , This is pretty accurate too .', 'In closing, this is a fantastic GPS with some very nice features and is very accurate in directions .']

    data = []

    with open("toy_input.txt", 'r') as toy:
        for line in toy:
            data.append(line)

    data = [s.strip() for s in data]

    summarizer = OpinosisSummarizer(
        data[:30], 2, 2, 4, 15, False, 0.5, False, False)
    '''print(summarizer._is_valid_start_node(("the", "DT")))
    print(summarizer._is_valid_end_node((".", ".")))
    sent = [('the', 'DT'), ('IPhone', 'NNP'), ('is', 'VBZ'), ('worth', 'JJ'), ('the', 'DT'), ('price', 'NN'), ('.', '.')]
    print(summarizer._is_valid_path(sent))
    print(summarizer._is_collapsible(('is', 'VBZ')))'''

    summary = summarizer.get_summary()

    for sent in summary:
        print(sent)
