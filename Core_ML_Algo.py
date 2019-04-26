
from sklearn.datasets import load_files
import nltk
import matplotlib.pyplot as plt
import numpy as np
import preProccessing as pP
import convert_text_to_numbers as cTN
import TFIDF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
#to asst me to download some data loke stopword or wordnet and so
#nltk.download('wordnet')
#nltk.download('stopwords')
#print(stopwords)
load_movies_data=load_files('./txt_sentoken')
#print(load_movies_data)
#set all text in x(data)
#set two targets in y(pos and neg)
text,lable=load_movies_data.data,load_movies_data.target
#print(text)
#print(y)
new_data=pP.preProccessing.data_filtaration(text)
#print(new_data)
#call text to vectorizer fun
text_vectors=cTN.text_to_vectorizer._vectorizer(new_data)
#calc TFIDF for text_vectors
tfidf=TFIDF.TFIDF._tfidf(text_vectors)
X=tfidf
#Training and Testing Sets (70% train data ,30% test data)
X_train,X_test,y_train,y_test=train_test_split(X,lable,test_size=0.3,random_state=0)

#we will Random Forest Algorithm to trian my model
#we should define number of trees as 1000( random forest)
classfier=RandomForestClassifier(n_estimators=1000,random_state=0)
classfier.fit(X_train,y_train)
y_predict=classfier.predict(X_test)

importances = classfier.feature_importances_

# evaluate the performance of a classification model 
print('\n===============================================================\n')
print('Confusion Matrix\n')
print(confusion_matrix(y_test,y_predict))  
conf_matrix=confusion_matrix(y_test,y_predict)
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
print(classification_report(y_test,y_predict))  
#print(accuracy_score(y_test, y_predict)) 



##Model visualization

std = np.std([tree.feature_importances_ for tree in classfier.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()



# Print the feature ranking
#print("Feature ranking:")
#
#for f in range(X.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#










#to save classfier model
#with open('text_classifier', 'wb') as picklefile:  
#    pickle.dump(classfier,picklefile)


#to save classfier model
#with open('text_classifier', 'rb') as training_model:  
#    model = pickle.load(training_model)
#    y_pred = model.predict(X_test)
#
#    print(confusion_matrix(y_test, y_pred))  
##    print(classification_report(y_test, y_pred))  
#    print(accuracy_score(y_test, y_pred)) 

