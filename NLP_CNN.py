import pandas as pd
import numpy as np
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# nltk.download('punkt')
# nltk.download('stopwords')


def clean(data, field):
    clean_tweet = []
    pat1 = '@[^ ]+'
    pat2 = '#[^ ]+'
    pat3 = 'www[^ ]+'
    pat4 = 'http[^ ]+'
    pat5 = '[0-9]'
    combined_pat = '|'.join((pat1, pat2, pat3, pat4, pat5))
    for t in data[field]:
        text = re.sub(combined_pat, '', t.lower())
        text = re.sub("n't", " not", text)
        text = re.sub(r'[^\w\s]', ' ', text)
        clean_tweet.append(text)
    return clean_tweet


def randomize_data(Data, columns, No_Data):
    np.random.seed(0)
    index = np.random.randint(0, 1599999, No_Data)
    Data = Data.loc[index, columns].reset_index(drop=True)
    return Data

def plot_Accuracy_Ep(Model):
    plt.plot(Model.history['acc'])
    plt.plot(Model.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_Loss_Ep(Model):
    plt.plot(Model.history['loss'])
    plt.plot(Model.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


######################## Naming COls ############################
cols = ['sentiment', 'id', 'date', 'query', 'user', 'tweet']
#################################################################

######################## Read CSV Data ##########################
Data = pd.read_csv("./Tweets-1.6M.csv", encoding='latin1', names=cols)
#################################################################

######################## Randomize Data #########################
Data = randomize_data(Data, ['tweet', 'sentiment'], 100000)
# print(Data.head())
#################################################################

######################## Clean Data #############################
Data['tweet'] = clean(Data, 'tweet')
#################################################################

######################## Replace 4 by 1 #########################
Data['sentiment'] = Data['sentiment'].replace(4, 1)
#################################################################

######################## putting Data in x & y ##################
x = Data['tweet']
y = Data['sentiment']
#################################################################

######################## Splitting data to test & train #########
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#################################################################

######################## Creating Vectorizer ####################
cv = CountVectorizer(ngram_range=(1, 3))  # ngram_range is the number of words
tfidf = TfidfVectorizer(ngram_range=(1, 3))
#################################################################

################## Tokenizing Train Data ########################
cv.fit(x_train)
tfidf.fit(x_train)
#################################################################

################## Transforming Train & Test Data to matrix######
x_train_cv = cv.transform(x_train)
x_test_cv = cv.transform(x_test)
x_train_tfidf = tfidf.transform(x_train)
x_test_tfidf = tfidf.transform(x_test)
# print(cv.get_feature_names())
# print(x_train_cv.toarray())
#################################################################

######################## Creating Logestic Regression ###########
lrcv = LogisticRegression()
lrtfidf = LogisticRegression()
#################################################################

######################## Fitting Logestic Regression ############
lrcv.fit(x_train_cv, y_train)  # must be of type dtype=np.float32
lrtfidf.fit(x_train_tfidf, y_train)  # must be of type dtype=np.float32
#################################################################

####################### Testing Logestic Regression ############
y_predict_cv = lrcv.predict(x_test_cv)
y_predict_tfidf = lrtfidf.predict(x_test_tfidf)
# print("y_predict_cv")
################################################################

####################### Getting Accuracy as percentage #########
print(" **********  Accuracy Using Logestic Regression  ********** ")
print('accuracy ', accuracy_score(y_test, y_predict_cv) * 100, '%')
print('accuracy ', accuracy_score(y_test, y_predict_tfidf) * 100, '%')
print("***********************************************************")
################################################################

####################### Postive & Negative sentances ###########
pos_df = Data.loc[Data['sentiment'] == 1]
neg_df = Data.loc[Data['sentiment'] == 0]
################################################################

############# Combining all Sentances into one paragrapgh ######
pos_df = pos_df['tweet'].str.cat(sep=' ')
neg_df = neg_df['tweet'].str.cat(sep=' ')
################################################################

############# Plotting the postive & Negative Words ############
plt.figure(figsize=(12, 10))
wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(pos_df)
plt.imshow(wordcloud)
plt.show()
#################################################################

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


################ Creating CNN Model #############################
model = Sequential()
model.add(Dense(units=32, input_dim=x_train_cv.shape[1]))
model.add(LeakyReLU(alpha=0.9))
model.add(Dropout(0.5))
model.add(Dense(units=32))
model.add(LeakyReLU(alpha=0.9))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())  # Param is the number of weights and bias = x_train_cv.shape[1] * 64 + 64
#################################################################

################ Fitting CNN Model ##############################
Model = model.fit(x_train_cv, y_train, epochs=5, batch_size=128, validation_data=(x_test_cv, y_test))
#################################################################

################ Plotting Accuracy vs Epoch #####################
plot_Accuracy_Ep(Model)
#################################################################

################ Plotting Loss vs Epoch #########################
plot_Loss_Ep(Model)
#################################################################