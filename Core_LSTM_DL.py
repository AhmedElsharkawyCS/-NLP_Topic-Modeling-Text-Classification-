import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential ,load_model
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns


classes={
          1:'spam',
          0:'ham'
        }

data= pd.read_csv('./Ham_Or_Spam_Text_Classification/spam.csv',delimiter=',',encoding='latin-1')
#print(data)
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
X = data.v2
Y = data.v1

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)


max_words = 1000
max_len = 150


tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)


sequences = tok.texts_to_sequences(X_train)
x_train = sequence.pad_sequences(sequences,maxlen=max_len)


test_sequences = tok.texts_to_sequences(X_test)
x_test = sequence.pad_sequences(test_sequences,maxlen=max_len)

#print(x_test)

#build LSTM deep learning model(model layers)
model = Sequential()
model.add(Embedding(max_words,50,input_length=max_len))
model.add(LSTM(64))
model.add(Dense(256))
model.add(Activation('relu'))
# Reduce Overfitting
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error',optimizer=RMSprop(),metrics=['accuracy'])
model.summary()
history =model.fit(x_train,Y_train,batch_size=128,epochs=10,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
##
##
###save lstm model
model.save('./Ham_Or_Spam_Text_Classification/LSTM_DeepLearningModel.h5')
# load our saved model
model = load_model('./Ham_Or_Spam_Text_Classification/LSTM_DeepLearningModel.h5')

##Evaluate the model on the test set.
accuracy=model.evaluate(x_test,Y_test)

#print('\nTest Result:')
#print('Loss: {:0.3f}'.format(accuracy[0]))
#print('Accuracy: {:0.3f}'.format(accuracy[1]))
print('\n===============================================================\n')
print('Confusion Matrix\n')
#to predict my our model and find confusion matrix
y_pred=model.predict(np.array(x_test))
y_pred=y_pred.round()
#print(y_pred)
print(confusion_matrix(np.array(Y_test).ravel(),np.array(y_pred).ravel()))
conf_matrix=confusion_matrix(np.array(Y_test).ravel(),np.array(y_pred).ravel())
print('\n===============================================================\n')
print('Model Evaluation \n')
Accuracy=((conf_matrix[0][0]+conf_matrix[1][1])/(conf_matrix[0][0]+conf_matrix[0][1]+conf_matrix[1][0]+conf_matrix[1][1]))*100
print('Accuracy: {:0.3f} %'.format(Accuracy.round()))
Precision=(conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1]))*100
print('Precision: {:0.3f} %'.format(Precision.round()))
Recall =(conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[1][0]))*100
print('Recall: {:0.3f} %'.format(Recall.round()))
#, F1 Score =  Mean(Precision, Recall) where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal
F1Score=2 * (Precision * Recall) / (Precision + Recall)
print('F1Score: {:0.3f} %'.format(F1Score.round()))
#Data visualization
sns.countplot(data.v1)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')
plt.show()
##Model visualization
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


















