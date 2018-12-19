
import nltk

y = ['the speed limit it the road you are on is not accurate.', 'the speed limit it the road you are on is not 100 % accurate.']


x1 = nltk.pos_tag(nltk.word_tokenize(y[0]))
x2 = nltk.pos_tag(nltk.word_tokenize(y[1]))

#x1 = set((('the', 'DT'), ('pedestrian', 'JJ'), ('mode', 'NN'), (',', ','), ('and', 'CC'), ('it', 'PRP'), ('is', 'VBZ'), ('amazing', 'JJ'), ('how', 'WRB'), ('accurate', 'JJ'), ('.', '.')))
#x2 = set((('the', 'DT'), ('pedestrian', 'JJ'), ('mode', 'NN'), (',', ','), ('it', 'PRP'), ('is', 'VBZ'), ('amazing', 'JJ'), ('how', 'WRB'), ('accurate', 'JJ'), ('.', '.')))


print(x1)
print(x2)

intersection_size = len(set(x1).intersection(set(x2)))
print(intersection_size)
union_size = len(set(x1).union(set(x2)))
print(union_size)

print(intersection_size/union_size)