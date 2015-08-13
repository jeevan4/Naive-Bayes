import collections
import numpy as np
import matplotlib.pyplot as plt
import sys

__author__ = 'Jeevan'

"""
Global variables initialization section
Loads data from the given files into the following Numpy Matrices
"""
train_data = np.loadtxt("train.data",int,skiprows=0)
test_data = np.loadtxt("test.data",int,skiprows=0)
test_label= np.loadtxt("test.label",int,skiprows=0)

# holds the data from train.label file in a list
label_list = []
# label_vocab_list = []
# contains the dictionary of total number of documents in each label
label_dict = {}
# contains the dictionary of probabilities for the occurrence of each label
label_prob = {}
# Matrix to store the MAP values
map_matrix = np.zeros((20,61188))
# Matrix to store the Classification
document_classification = np.zeros((7505,20))
# Matrix to hold the counts of each word present in the test.data
vocab_occurance = np.zeros((61188,7505),dtype=np.int)
# Matrix to hold the label for the highest probability for each document
argmax_matrix = np.zeros((7505,1),dtype=np.int)
# Holds the confusion matrix for the given and predicted labels
confusion_matrix = np.zeros((20,20),dtype=np.int)

"""
    Function to calculate the Likelihood probabilities
    for each label in the given set of documents.
"""
def mla_calc():
    global label_list,label_dict,label_prob,map_matrix,vocab_occurance,document_classification,argmax_matrix,confusion_matrix
    global train_data,test_data,test_label
    with open("train.label") as f:
        for index, all_lines in enumerate(f):
            # label_prob.setdefault(index,int(all_lines.replace('\n','')))
            label_list.append(int(all_lines.replace('\n','')))
            # z = np.array([index+1,all_lines])
        # print(label_list)
        from collections import Counter
        label_dict = Counter(label_list)
        # print(label_dict)
        # label_dict = sorted(label_dict.items(),key=itemgetter(0))
        labelLen = 11269
        labelTot = len(label_dict)
        for i in range(1,labelTot+1):
            label_prob.setdefault(i,round(label_dict.get(i)/labelLen,5))
        # print(label_prob)
    return

"""
    :param beta_arg:
    :return:accuracy
    This function takes an argument for Beta value
    and calculates the probability of occurrence of each word for a given label.
    Initializes them into map_matrix, returns accuracy from classify()
"""

def map_calc(beta_arg):
    vocabLen = 61188
    beta=beta_arg
    alpha=1+beta
    # print(sorted(label_dict.keys()))
    # train_data = np.loadtxt("D:/Sem-2/machine/project2/data/data/train.data",int,skiprows=0)
    start = 0
    for i in range(1,21):
        endoc = label_dict.get(i) + start
        # print(endoc)
        total_words = train_data[np.logical_and(train_data[:,0] > start,train_data[:,0] <= endoc)][:,np.array([False,False,True])]
        # print(i,sum(total_words))
        denominator = (sum(total_words)+((alpha-1)*vocabLen))
        vocab_words_total = train_data[np.logical_and(train_data[:,0] > start,train_data[:,0] <= endoc)][:,np.array([False,True,True])]
        parse = set(list(vocab_words_total[:,0]))
        # print(parse)
        numerator_zero=(alpha-1)
        for arg in range(1,vocabLen+1) :
            numerator=np.sum(vocab_words_total[vocab_words_total[:,0] == arg][:,np.array([False,True])])+(alpha-1)
            map_val = (numerator/denominator)
            # print(i,arg,numerator/denominator)
            map_matrix[i-1,arg-1]=map_val
            # print(arg,sum1)
        start = endoc
        # map_matrix[np.where(map_matrix==0)]=(numerator_zero/denominator)

    accuracy=classify()
    return  accuracy
    # print(vocab_words_total[:,0:2])

"""
    :return: accuracy
    The function applies the log2 to the map_matrix and classifies
    the test data based on the arg max of MLE and MAP values. Then
    compares the predicted labels with the given test labels to obtain
    accuracy and confusion matrix.

"""
def classify():

    map_matrix_log=(np.log2(map_matrix)).T
    # test_data = np.loadtxt("D:/Sem-2/machine/project2/data/data/test.data",int,skiprows=0)
    total_documents=set(list(test_data[:,0]))
    for doc in total_documents:
        doc_split=test_data[test_data[:,0] == doc][:,np.array([False,True,True])]
        vocab_count=list(doc_split[:,0])
        for vocab in vocab_count:
            vocab_occurance[vocab-1,doc-1]=doc_split[doc_split[:,0] == vocab][:,np.array([False,True])]
    document_classification=np.dot(vocab_occurance.T,map_matrix_log)
    for label in range(0,20):
        if(label_prob.get(label+1) != None):
            # print(label_prob.get(label+1))
            document_classification[:,label] = document_classification[:,label]+np.log2(label_prob.get(label+1))
    argmax_matrix = np.argmax(document_classification,axis=1)+1
    # print(argmax_matrix[:,])
    # test_label= np.loadtxt("D:/Sem-2/machine/project2/data/data/test.label",int,skiprows=0)
    accurate=0
    for arg in range(0,7504):
        if argmax_matrix[arg] == test_label[arg]:
            accurate=accurate+1
    accuracy = (accurate*100/7505)
    print("Accuracy obtained is : ",accuracy)
    # print("different items",np.setdiffId(argmax_matrix,test_data,True).size)
    from collections import Counter
    label_counts=Counter(test_label)
    begin = 0
    for i in range(1,21):
        ending=label_counts.get(i)+begin
        estimated_label_counts=Counter(argmax_matrix[begin:ending])
        for j in estimated_label_counts.keys():
            confusion_matrix[i-1,j-1]=estimated_label_counts.get(j)
        print("Confusion Percentage for label ",i,(np.amax(confusion_matrix[i-1,:])*100)/label_counts.get(i))
        begin=ending
    print("==========================Confusion-Matrix==================================")
    print(confusion_matrix)
    return accuracy

"""
    The function calculates uses the Bayes Theorem to calculate Posteriors
    based on Priors and Likelihood. Then, calculates the sum of probabilities
    for all labels given a word. By sorting these probabilities in ascending order
    we can give better rank to low frequency words.
"""
def vocab_rank():
    vocab_rank = {}
    vocab_data = np.genfromtxt("D:/Sem-2/machine/project2/data/data/vocabulary.txt",dtype='str')
    vocab_matrix = np.zeros((20,61188))
    for i in range(0,20):
        vocab_matrix[i,:] = map_matrix[i,:]*label_prob.get(i+1)
    for arg in range(0,61188):
        vocab_rank.setdefault(vocab_data[arg],np.sum(vocab_matrix[:,arg]))
    from collections import OrderedDict
    sorted_vocab=OrderedDict(sorted(vocab_rank.items(),key=lambda a:a[1])[:100])
    print("Top 100 words with highest measure :",list(sorted_vocab.keys()))

if __name__ == '__main__':

    print("Calculating Accuracy and Confusion Matrix for Beta = (1/|V|)")
    mla_calc()
    map_calc(round((1/61188),5))
    vocab_rank()
    beta = input('Enter the Desired Beta Value to calculate Accuracy and Confusion Matrix again [Hit Enter to Exit]:')
    if not beta:
        sys.exit()
    map_calc(float(beta))

    # below part is commented after plotting graph
    # accurate=[]
    # count = 0
    #
    # for i in beta:
    #     beta=[0.00001,0.0003,0.0006,0.0009,0.001,0.003,0.006,0.009,0.01,0.04,0.07,0.09,0.1,0.3,0.8,1]
    #     print("Calculating Accuracy for Beta = ",i)
    #     accurate.append(map_calc (i))
    #     print(accurate[count])
    #     count += 1
    # plt.subplot(111)
    # plt.semilogx(beta, accurate)
    # plt.title('Graph for the given Beta value range')
    # plt.xlabel('Beta Values')
    # plt.ylabel('Accuracy Percentage')
    # plt.grid(True)
    # plt.show()
