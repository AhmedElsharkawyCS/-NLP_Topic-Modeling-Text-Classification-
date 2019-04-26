#to proccess the incoming data 
from nltk.stem import WordNetLemmatizer
import re
stemmer=WordNetLemmatizer()

class preProccessing:
    def data_filtaration(data):
        documents=[]
        for sen in range(0,len(data)):
#            print(sen)
#            Remove all the special characters and get all text as tokens
            document=re.sub(r'\W',' ',str(data[sen]))
#            print(document)
                # remove all single characters(ahmed's to ahmed)
            document=re.sub(r'\s+[a-zA-Z]\s+',' ', document)
             # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
#            print(document)
#            replace all double spacing with single space (ignore it from the string)
            document=re.sub(r'\s+',' ',document,flags=re.I)
#            print(re.I)
#            remove the prefix 'b' if we use  bytes format but i we can use it to make sure
            document=re.sub(r'^b\s+','',document)
#            into lower case 
            document=document.lower()
            
#            Lemmatization
            #split all text into tokens
            document=document.split()
#            print(document)
            #conver every word to basic word or main word(reduce the word into dictionary root form)
            document=[stemmer.lemmatize(word) for word in document]
            #will put aingle space between the words
            document=' '.join(document)
#            print(document)
            documents.append(document)
        return documents    