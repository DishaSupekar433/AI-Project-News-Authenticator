#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import modules

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle


# In[3]:


#locate dataset and store it in a  variable

dataset_path = 'C:\\Users\\DISHA\\Downloads\\News Authenticator\\train.csv'
df = pd.read_csv(dataset_path)


# In[4]:


#read top 10 values

df.head(10)


# In[5]:


#read dataset description

df.describe()


# In[6]:


#get dataset info

df.info()


# In[7]:


#to find no. of null values of each columns

df.isnull().sum()


# In[8]:


#to fill the null value with one empty string

df=df.fillna('')


# In[9]:


#now all the null values are filled

df.isnull().sum()


# In[10]:


#to see all the columns

df.columns


# In[11]:


#drop useless columns and keep columns like text and label from axis no. 1

df=df.drop(['id', 'title', 'author'],axis=1)


# In[12]:


#to verify if iit worked or not

df.head()


# In[17]:


#

port_stem=PorterStemmer()


# In[18]:


port_stem


# In[19]:


port_stem.stem("Hi * % %@@@")


# In[20]:


def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con


# In[22]:


stemming('Hi this is Disha')


# In[24]:


stemmer = SnowballStemmer(language='english')


# In[25]:


#takes time

df['text'] = df['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))


# In[26]:


x=df['text']


# In[27]:


y=df['label']


# In[28]:


y.shape


# In[30]:


x_train , x_test , y_train, y_test = train_test_split(x, y, test_size=0.20)


# In[32]:


vect=TfidfVectorizer()


# In[33]:


x_train=vect.fit_transform(x_train)
x_test=vect.transform(x_test)


# In[34]:


x_test.shape


# In[36]:


model=DecisionTreeClassifier()


# In[37]:


model.fit(x_train, y_train)


# In[38]:


prediction=model.predict(x_test)


# In[39]:


prediction


# In[40]:


model.score(x_test, y_test)


# In[42]:


pickle.dump(vect, open('vector.pkl', 'wb'))


# In[43]:


pickle.dump(model, open('model.pkl', 'wb'))


# In[44]:


vector_form=pickle.load(open('vector.pkl', 'rb'))


# In[45]:


load_model=pickle.load(open('model.pkl', 'rb'))


# In[46]:


def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction


# In[47]:


val=fake_news("""In these trying times, Jackie Mason is the Voice of Reason. [In this week’s exclusive clip for Breitbart News, Jackie discusses the looming threat of North Korea, and explains how President Donald Trump could win the support of the Hollywood left if the U. S. needs to strike first.  “If he decides to bomb them, the whole country will be behind him, because everybody will realize he had no choice and that was the only thing to do,” Jackie says. “Except the Hollywood left. They’ll get nauseous. ” “[Trump] could win the left over, they’ll fall in love with him in a minute. If he bombed them for a better reason,” Jackie explains. “Like if they have no transgender toilets. ” Jackie also says it’s no surprise that Hollywood celebrities didn’t support Trump’s strike on a Syrian airfield this month. “They were infuriated,” he says. “Because it might only save lives. That doesn’t mean anything to them. If it only saved the environment, or climate change! They’d be the happiest people in the world. ” Still, Jackie says he’s got nothing against Hollywood celebs. They’ve got a tough life in this country. Watch Jackie’s latest clip above.   Follow Daniel Nussbaum on Twitter: @dznussbaum """)


# In[48]:


if val==[0]:
    print('reliable')
else:
    print('unreliable')


# In[ ]:




