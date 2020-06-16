#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle


# In[14]:


app=Flask(__name__,template_folder='template')
model=pickle.load(open('model.pkl','rb'))


# In[15]:


@app.route('/')
def home():
    return render_template('index.html')


# In[16]:


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x)for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0],2)

    return render_template('index.html',predict_text='Employee salary should be {}'.format(output))


# In[17]:


if __name__=="__main__":
    app.run(debug=True)


# In[ ]:




