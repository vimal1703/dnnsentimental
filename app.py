#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle



app = Flask(__name__ , templates_folder = templates)


@app.route('/')
def home():
    return render_template('sentiment.html')


@app.route('/sentiment.html', methods=['POST', 'GET'])
def sentiment():
    
    str_features = [str(x) for x in request.form.values()]
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(str_features))))['compound']
    if(score >0):
        label = 'This sentence is positive'

    elif(score ==0):
        label = 'This sentence is neutral'

    else: 
        label = 'This sentence is negative'
    output = label
    return render_template('sentiment.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




