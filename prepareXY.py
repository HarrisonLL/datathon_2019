import json
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
from nltk import tokenize
nltk.download('stopwords')

punctuation = [',','.',':','(',')','!','?','"','“','”']
X = {'BAYZF':{}, 'HON':{}, 'MMM':{}, 'SYF':{}}
Y = {'BAYZF':{}, 'HON':{}, 'MMM':{}, 'SYF':{}}
json_filelist = ['8k.json']
csv_filelist = ['BAYZF.csv','HON.csv','MMM.csv','SYF.csv']
longterm = 40
midterm = 20
shortterm = 7

def split_text(text):
    return tokenize.sent_tokenize(text)

def delete_stop_words(sentence):
    stop_words = set(stopwords.words('english'))       
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]       
    filtered_sentence = []       
    for w in word_tokens: 
        if w not in stop_words and w not in punctuation and w not in string.punctuation: 
            filtered_sentence.append(w) 
    new_sentence = ""
    for i in range(len(filtered_sentence)):
        if i != len(filtered_sentence) - 1:
            new_sentence += (filtered_sentence[i] + ' ')
        else:
            new_sentence += (filtered_sentence[i] + '.')     
    return new_sentence

def cleanX(text):
    cleanedText = ""
    sentences_in_x = split_text(text)
    for sentence in sentences_in_x:
        cleanedText += delete_stop_words(sentence)
    return cleanedText

def get_X(filename):
    with open(filename,'r') as f1:
        for line in f1:
            dictionary = json.loads(line)
            CompName = dictionary['CompanyName']
            date = dictionary['Date']
            Text= cleanX(dictionary['Text'])
            if date not in X[CompName].keys():
                X[CompName][date] = ""
                X[CompName][date] += Text
            else:
                X[CompName][date] += Text
    f1.close()

def SLR(x,y):
    x = np.array(x)
    y = np.array(y)
    n = np.size(x)  
    m_x, m_y = np.mean(x), np.mean(y) 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return b_1

def get_Y(filename):
    CompName = filename[ : filename.find('.')]
    df = pd.read_csv(filename)
    length = df.shape[0]
    for i in range(length):
        if i <= length - longterm:
            long_x = list(range(0,longterm))
            long_y = []
            for j in range(longterm):
                long_y.append(df.iloc[i+j]["Close"])
            longrun = SLR(long_x, long_y)
            mid_x = list(range(0,midterm))
            mid_y = long_y[:midterm]
            midrun = SLR(mid_x, mid_y)
            short_x = list(range(0,shortterm))
            short_y = long_y[:shortterm]
            shortrun = SLR(short_x, short_y)
            date = df.iloc[i]["Date"]
            Y[CompName][date] = {}
            Y[CompName][date]['ST'] = shortrun
            Y[CompName][date]['MT'] = midrun
            Y[CompName][date]['LT'] = longrun
        elif (i > length - longterm) and (i <= length - midterm):
            long_x = list(range(0,length - i))
            long_y = []
            for j in range(length - i):
                long_y.append(df.iloc[i+j]["Close"])
            longrun = SLR(long_x, long_y)
            mid_x = list(range(0,midterm))
            mid_y = long_y[:midterm]
            midrun = SLR(mid_x, mid_y)
            short_x = list(range(0,shortterm))
            short_y = long_y[:shortterm]
            shortrun = SLR(short_x, short_y)
            date = df.iloc[i]["Date"]
            Y[CompName][date] = {}
            Y[CompName][date]['ST'] = shortrun
            Y[CompName][date]['MT'] = midrun
            Y[CompName][date]['LT'] = longrun
        elif (i > length - midterm) and (i <= length - shortterm):
            long_x = list(range(0,length - i))
            long_y = []
            for j in range(length - i):
                long_y.append(df.iloc[i+j]["Close"])
            longrun = SLR(long_x, long_y)
            midrun = longrun
            short_x = list(range(0,shortterm))
            short_y = long_y[:shortterm]
            shortrun = SLR(short_x, short_y)
            date = df.iloc[i]["Date"]
            Y[CompName][date] = {}
            Y[CompName][date]['ST'] = shortrun
            Y[CompName][date]['MT'] = midrun
            Y[CompName][date]['LT'] = longrun            
        else:
            long_x = list(range(0,length - i))
            long_y = []
            for j in range(length - i):
                long_y.append(df.iloc[i+j]["Close"])
            longrun = SLR(long_x, long_y)
            midrun = longrun
            shortrun = longrun 
            date = df.iloc[i]["Date"]
            Y[CompName][date] = {}
            Y[CompName][date]['ST'] = shortrun
            Y[CompName][date]['MT'] = midrun
            Y[CompName][date]['LT'] = longrun

for filename in json_filelist:
    get_X(filename)

for filename in csv_filelist:
    get_Y(filename)

with open('X_train.json', 'a') as f1:
    json.dump(X, f1)
    f1.write('\n')
f1.close()

with open('Y_train.json', 'a') as f2:
    json.dump(Y, f2)
    f2.write('\n')
f2.close()