from os import listdir
import codecs
from sum_basic import SumBasic
from opinosis_summarizer import OpinosisSummarizer

'''
Opinosis dataset provided by: https://github.com/kavgan/opinosis/blob/master/opinosis-dataset-documentation.pdf

To Run: 
Ensure 
1) FULL_DOCS_FOLDER points to the directory containing the full article/text 
2) SYSTEM_SUMMARIES_FOLDER points to directory for summaries to be written to
'''

FULL_DOCS_FOLDER = 'datasets/full_docs/'
SYSTEM_SUMMARIES_FOLDER = 'datasets/system_summaries/'

full_doc_files= listdir(FULL_DOCS_FOLDER)

for doc in full_doc_files:
    input_text = []

    with codecs.open(FULL_DOCS_FOLDER+doc, 'r', encoding='utf-8', errors='ignore') as text:
        for line in text:
            input_text.append(line)

    #sumbasic = SumBasic(input_text)

    #prep = sumbasic.get_summary(15)

    opinosis = OpinosisSummarizer(input_text,2,2,4,15, False, 0.5)

    summary = opinosis.get_summary()

    file = open(SYSTEM_SUMMARIES_FOLDER+doc[:-5], "w") #remove ".data" ending

    for sentence in summary:
        file.write(sentence + '\n')
    file.close()