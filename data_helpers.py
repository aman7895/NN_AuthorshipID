
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


# In[2]:

# This function is called from Main and expects train and test values for x and y
def load_ag_data(authors = None, docID = None): 
    
    data = pd.read_csv("aman_ml_authors_10.csv")
    
    print(data.shape)
    
    authorList = data.author_id.unique()
    
    authorListfull = data.author_id.tolist()
    
    labels = []
    
    textlist = data.drop('author_id', axis = 1)
    
    print(textlist.shape)
    
    textlist = textlist.to_records(index = False).tolist()
    
    print(len(textlist[0]))
    
    print(len(textlist))
    
    for auth in authorListfull:
        labels = labels + [authorList.tolist().index(auth)]
        
    labels = to_categorical(np.asarray(labels))
    
    X_train, X_test, y_train, y_test = train_test_split(textlist, labels, test_size=0.3, random_state=123)
    
    # return (texts, labels, labels_index, samples)

    return ((X_train, y_train), (X_test, y_test), authorList)
