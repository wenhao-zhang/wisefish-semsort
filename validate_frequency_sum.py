import rougescore
from os import listdir
import codecs

''' 
example of the format required for reference and model 

summary= ['this', "is", 'an', 'example', '.']
There can only be one summary 

references = [['this', "is", 'another', 'example', '.'], [...], [...]]
there can be multiple reference/gold standards.


---- Test Explanation of the Rouge Score Usage ----

recall: shared words/#words in reference(gold standard)
precision: shared words/#words in the summary(machine generated)

in our toy example below : 
recall: 5/5 = 1.0
precision: 5/6 = 0.833...

Rouge-N, where N is the n-gram size. ie: 1 = unigram, 2-bigram 

summary_test = ["the", "dog", "ate", "all", "my", "homework"]
reference_test = [["the", "dog", "ate", "my", "homework"]]
rouge1_recall_test = rougescore.rouge_1(summary_test, reference_test, favor_recall)
rouge1_precision_test = rougescore.rouge_1(summary_test, reference_test, favor_precision)

print(summary_test)
print(reference_test)
print("Rouge-1 Recall: = " + str(rouge1_recall_test))
print("Rouge-1 Precision:  = " + str(rouge1_precision_test))

'''

SYSTEM_DIR = 'datasets/system_summaries/'
MODEL_DIR = 'datasets/model_summaries/'

list_of_model_names = listdir(MODEL_DIR)

sum_rouge1_precision, sum_rouge2_precision, sum_rouge1_recall, sum_rouge2_recall = (0,)*4

sum_summary_text = list()
sum_reference_text = list()
sum_num_references = 0
sum_num_summaries = len(list_of_model_names)

for name in list_of_model_names:
    summary_text = ""
    with codecs.open(SYSTEM_DIR+name+'.txt', 'r', encoding='utf-8', errors='ignore') as text:
        for line in text:
            summary_text += line
            sum_summary_text.append(line.split())  # Append ["word1","word2" ...]
    summary = summary_text.split()

    num_models = len(listdir(MODEL_DIR+name)) #There are between 4 and 5 gold standards for each topic
    references = list()
    reference_text = ["" for x in range(num_models)]
    sum_num_references += num_models

    # Model number in files is offset by 1 eg: file.1.gold through file.5.gold
    for model_num in range(num_models):
        with codecs.open(MODEL_DIR+name+'/'+name+'.' + str(model_num+1) + '.gold', 'r', encoding='utf-8', errors='ignore') as text:
            for line in text:
                reference_text[model_num] += line
                sum_reference_text.append(line.split())

    # Assemble gold standard summaries into seperated lists
    for ref in reference_text:
        references.append(ref.split())
    print("\n--- System summary: ---")
    print(summary)
    print("\n--- Gold standard references: ---")
    print(references)

    favor_recall, favor_precision = (0,1) #rouge score take alpha value from 0 (recall) to 1(precision)

    rouge1_recall = rougescore.rouge_1(summary, references, favor_recall)
    rouge2_recall = rougescore.rouge_2(summary, references, favor_recall)
    rouge1_precision = rougescore.rouge_1(summary, references, favor_precision)
    rouge2_precision = rougescore.rouge_2(summary, references, favor_precision)

    print("\n--- Results --- ")
    print("Rouge-1 recall score: " + str(rouge1_recall))
    print("Rouge-2 recall score: " + str(rouge2_recall))

    print("Rouge-1 precision score: " + str(rouge1_precision))
    print("Rouge-2 precision score: " + str(rouge2_precision))

    sum_rouge1_precision += rouge1_recall
    sum_rouge2_precision += rouge2_recall

    sum_rouge1_recall += rouge1_precision
    sum_rouge2_recall += rouge2_precision

print("\n\n --- Final Results --- ")
print("rouge-1 precision: " + str(sum_rouge1_precision/len(list_of_model_names)))
print("rouge-2 precision: " + str(sum_rouge2_precision/len(list_of_model_names)))
print("rouge-1 recall: " + str(sum_rouge1_recall/len(list_of_model_names)))
print("rouge-2 recall: " + str(sum_rouge2_recall/len(list_of_model_names)))

summary_word_count = 0
for blob in sum_summary_text:
    summary_word_count += len(blob)

reference_word_count = 0
for blob in sum_reference_text:
    reference_word_count += len(blob)

print("number of words / generated summaries: " + str(summary_word_count/sum_num_summaries))
print("number of words / gold standard models: " + str(reference_word_count/sum_num_references))

