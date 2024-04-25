import streamlit as st 
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer 

# Load model and vectorizer
vector_form = pickle.load(open(r'C:\Users\DISHA\Downloads\News Authenticator\vector.pkl', 'rb'))
load_model = pickle.load(open(r'C:\Users\DISHA\Downloads\News Authenticator\model.pkl', 'rb'))


# Initialize NLTK components
port_stem = PorterStemmer()

def stemming(content):
    # Perform stemming and text preprocessing
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con

def fake_news(news):
    # Preprocess news content and make prediction
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction[0]  # Return the prediction result

# Streamlit app UI
def main():
    st.title('Fake News Classification App')
    st.subheader('Enter the news content below to classify it as reliable or unreliable')
    sentence = st.text_area("Enter your news content here", "", height=200)
    if st.button("Predict"):
        prediction_class = fake_news(sentence)
        if prediction_class == 0:
            st.success('The news is classified as Reliable')
        else:
            st.warning('The news is classified as Unreliable')

if __name__ == '__main__':
    main() 
