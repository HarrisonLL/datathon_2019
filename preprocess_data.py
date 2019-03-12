import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import string
import json


def remove_stopwords(text):
    stop_words = set(stopwords.words('english')) 
      
    #word_tokens = word_tokenize(text) 
      
    filtered_sentence = [w for w in text if not w in stop_words] 
      
    filtered_sentence = [] 
      
    for w in text: 
        if w.lower() not in stop_words: 
            filtered_sentence.append(w.lower())
    no_stop_words_text = ""
    for item in filtered_sentence:
        no_stop_words_text += item
        no_stop_words_text += ' '
    return no_stop_words_text

def remove_string_special_characters(s):
    """
    remove special characters from within a string
    parameters: s(str): single input string
    return: stripped(str): A string with special characters removed
    """
    # replace special character with ''
    stripped = re.sub('[^\w\s]','',s)
    stripped = re.sub('_','',stripped)

    # change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)
    # remove start and end white spaces
    stripped = stripped.strip()

    return stripped

def get_doc(text_sents_clean):
    """
    This function splits the text into sentences and considering
    each sentence as a document, calculate the total word count of each.
    """
    doc_info = []
    i = 0
    for sent in text_sents_clean:
        i += 1
        count = count_words(sent)
        temp = {'doc_id' : i, 'doc_length' : count}
        doc_info.append(temp)
    return doc_info

def count_words(sent):
    """
    This function returns the total number of words in the input text
    """
    count = 0
    words = word_tokenize(sent)
    for word in words:
        count += 1
    return count

def create_freq_dict(sents):
    """
    This function creates a frequency dictionary
    for each word in each document
    """
    i = 0
    freqDict_list = []
    for sent in sents:
        i += 1
        freq_dict = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            if word in freq_dict:
                freq_dict[word] += 1
            else:
                freq_dict[word] = 1
            temp = {'doc_id': i, 'freq_dict': freq_dict}
        freqDict_list.append(temp)
    return freqDict_list

def computeTF(doc_info, freqDict_list):
    """
    tf = (frequency of the term in the doc/total number of terms in the doc)
    """
    TF_scores = []
    for tempDict in freqDict_list:
        idd = tempDict['doc_id']
        for k in tempDict['freq_dict']:
            temp = {'doc_id': idd,
                    'TF_score': tempDict['freq_dict'][k]/doc_info[idd-1]['doc_length'],
                    'key': k}
            TF_scores.append(temp)
    return TF_scores

def computeIDF(doc_info, freqDict_list):
    """
    idf = ln(total num of docs / number of docs with term in it)
    """
    IDF_scores = []
    counter = 0
    for dic in freqDict_list:
        counter += 1
        for k in dic['freq_dict'].keys():
            count = sum([k in tempDict['freq_dict'] for tempDict in freqDict_list])
            temp = {'doc_id': counter, 'IDF_score': math.log(len(doc_info)/count),'key':k}

            IDF_scores.append(temp)
    return IDF_scores

def computeTFIDF(TF_scores, IDF_scores):
    TFIDF_scores = []
    for j in IDF_scores:
        for i in TF_scores:
            if j['key'] == i['key'] and j['doc_id'] == i['doc_id']:
                temp = {'doc_id': j['doc_id'],
                        'TFIDF_score': j['IDF_score']*i['TF_score'],
                        'key': i['key']}
        TFIDF_scores.append(temp)
    return TFIDF_scores

def read_data(filename):
    with open(filename, "r") as f:
        text = f.read()
    return text
###############need to fill
filelist = ['mmm_doc_txts October 2, 2017    .txt','mmm_doc_txts October 6, 2017    .txt']

def get_hottest_words(filename):
    hi = read_data(filename)
    hii = word_tokenize(hi)

    additional_punctuation = ['-','--',',','.','*','，','。']
    digit = ['0','1','2','3','4','5','6','7','8','9']

    text1 = []
    for word in hii:
        check = 0
        for element in word:
            if (element in string.punctuation) or (element in additional_punctuation) or (element in digit):
                check += 1
        if check == 0:
            text1.append(word)

    text1 = remove_stopwords(text1)
    text_sents = sent_tokenize(text1)
    text_sents_clean = [remove_string_special_characters(s) for s in text_sents]
    doc_info = get_doc(text_sents_clean)

    freqDict_list = create_freq_dict(text_sents_clean)
    freqDict = freqDict_list[0]['freq_dict']
    TF_scores = computeTF(doc_info, freqDict_list)
    IDF_scores = computeIDF(doc_info, freqDict_list)

    freqlist = sorted(freqDict, key = freqDict.get, reverse = True)
    top_10_hot_words = freqlist[:20]
    storage = {}
    storage[filename] = top_10_hot_words
    with open('hottest_words.json', 'a') as outfile:  
        json.dump(storage, outfile)
        outfile.write('\n')
    outfile.close()

for filename in filelist:
    get_hottest_words(filename)