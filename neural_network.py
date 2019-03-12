import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
import csv
import datetime
from nltk import tokenize
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM



nltk.download('stopwords')

wordList = np.load('wordList.npy')
wordList = wordList.tolist()
print('Loaded the word list!')
wordVectors = np.load('vectorList.npy')

maxSequence = 100
numDimension = 50

punctuation = [',','.',':','(',')','!','?','"','“','”']
# text 以天为单位



def delete_stop_words(text): # this text is the training data for a certain day
    stop_words = set(stopwords.words('english'))       
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]       
    filtered_text = []       
    for w in word_tokens: 
        if w not in stop_words and w not in punctuation: 
            filtered_text.append(w) 
    return filtered_text

def construct_big_sentence(filtered_text):
    big_sentence_bag = [] # text for 1 day, a list of list, each sublist has length 200
    counter = 0
    for word in filtered_text:
        counter += 1
        if counter == maxSequence:
            big_sentence = filtered_text[:maxSequence]
            big_sentence_bag.append(big_sentence)
        elif counter % maxSequence == 0 and counter != len(filtered_text):
            start = (int(counter / maxSequence) - 1) * maxSequence
            big_sentence = filtered_text[start : start + maxSequence]
            big_sentence_bag.append(big_sentence)
        elif counter == len(filtered_text):
            start = int(counter / maxSequence) * maxSequence
            big_sentence = filtered_text[start : ]
            big_sentence_bag.append(big_sentence)    
    return (big_sentence_bag)

def sentence_index(big_sentence_bag):
    big_sentence_bag_index = []
    for big_sentence in big_sentence_bag:
        big_sentence_index = []
        for word in big_sentence:
            if word in wordList:
                wordindex = wordList.index(word.lower())
                big_sentence_index.append(wordindex)
            else:
                pass
        big_sentence_bag_index.append(big_sentence_index)
    return big_sentence_bag_index

def everyday_sentence_matrix_bag(big_sentence_bag_index):
    daily_sentence_matrix_bag = []
    for shortText in big_sentence_bag_index:
        shortTextMatrix = []
        if len(shortText) == maxSequence:            
            for word_index in shortText:
                shortTextMatrix.append(wordVectors[word_index])
        else:
            zero = [0 for i in range(numDimension)]
            diff = maxSequence - len(shortText)
            for word_index in shortText:
                shortTextMatrix.append(wordVectors[word_index])
            for i in range(diff):
                shortTextMatrix.append(zero)
        daily_sentence_matrix_bag.append(shortTextMatrix)
    return np.array(daily_sentence_matrix_bag)
################ 代码调试区域
def isLineEmpty(line):
    return len(line.strip()) == 0

def read_text(file):
    text = ""
    new_text = ""
    with open (file,'r') as f:
        for line in f:
            if isLineEmpty(line) == False:
                text+=line
    f.close()
    print(text)
    sentences = tokenize.sent_tokenize(text)
    for sentence in sentences:
        new_text += (sentence + ' ')
    return new_text

def get_them(text):
    filtered_text = delete_stop_words(text)
    big_sentence_bag = construct_big_sentence(filtered_text)
    big_sentence_bag_index = sentence_index(big_sentence_bag)
    daily_sentence_matrix_bag = everyday_sentence_matrix_bag(big_sentence_bag_index)
    return daily_sentence_matrix_bag

with open('X_train.json') as json_file:  
    raw_x = json.load(json_file)
json_file.close()

with open('Y_train.json') as json_file:
    Y_train = json.load(json_file)
json_file.close()

X_train = raw_x
for key_comp_name in raw_x.keys():
    for key_date in raw_x[key_comp_name].keys():
        n_d_array = get_them(raw_x[key_comp_name][key_date])
        X_train[key_comp_name][key_date] = n_d_array

real_x_train = []
real_y_train_st= []
real_y_train_mt= []
real_y_train_lt= []
for key_comp_name in X_train.keys():
    for key_date in X_train[key_comp_name].keys():
        #if X_train[key_comp_name][key_date].shape[0]>1:
        for i in range(X_train[key_comp_name][key_date].shape[0]):
            real_x_train.append(X_train[key_comp_name][key_date][i])
            real_y_train_st.append(Y_train[key_comp_name][key_date]['ST'])
            real_y_train_mt.append(Y_train[key_comp_name][key_date]['MT'])
            real_y_train_lt.append(Y_train[key_comp_name][key_date]['LT'])
# create and fit the LSTM network



model1 = Sequential()
model1.add(LSTM(units=250, return_sequences=True, input_shape=(100,50)))
model1.add(Dropout(0.2))
model1.add(LSTM(units=250))
model1.add(Dropout(0.2))
model1.add(Dense(1))
print('hi')
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit(real_x_train, real_y_train_st, epochs=10, batch_size=10, verbose=2)


model2 = Sequential()
model2.add(LSTM(units=250, return_sequences=True, input_shape=(100,50)))
model2.add(Dropout(0.2))
model2.add(LSTM(units=250))
model2.add(Dropout(0.2))
model2.add(Dense(1))
print('hi')
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(real_x_train, real_y_train_mt, epochs=10, batch_size=10, verbose=2)



model3 = Sequential()
model3.add(LSTM(units=250, return_sequences=True, input_shape=(100,50)))
model3.add(Dropout(0.2))
model3.add(LSTM(units=250))
model3.add(Dropout(0.2))
model3.add(Dense(1))

print('hi')
model3.compile(loss='mean_squared_error', optimizer='adam')
model3.fit(real_x_train, real_y_train_lt, epochs=10, batch_size=10, verbose=2)


x_test = real_x_train[-100:]
y_test_st = real_y_train_st[-100:]
y_test_mt = real_y_train_st[-100:]
y_test_lt = real_y_train_st[-100:]


# inputs = new_data.values
# print(inputs)
# print(x_train.shape[0])
# print((x_train.shape[0],len(inputs)-60-timeFrame))
# X_test = []
# for i in range(x_train.shape[0],len(inputs)-60-timeFrame):
#     X_test.append(inputs[i:i+60,0])
# X_test = np.array(X_test)
# print(X_test.shape)
# X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
# print(X_test.shape)

y_st = model1.predict(x_test)
y_mt = model2.predict(x_test)
y_lt = model3.predict(x_test)


print(y_st)
print(y_mt)
print(y_lt)







