from flask import Flask, render_template, request, session
from flask_sqlalchemy import SQLAlchemy
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import mysql.connector
import pandas as pd
import random

# Load model and vectorizer
vector_form = pickle.load(open("vector.pkl", "rb"))
load_model = pickle.load(open("model.pkl", "rb"))

# Initialize NLTK components
port_stem = PorterStemmer()

def stemming(content):
    # Perform stemming and text preprocessing
    con = re.sub("[^a-zA-Z]", " ", content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words("english")]
    con = " ".join(con)
    return con

def fake_news(news):
    # Preprocess news content and make prediction
    news = stemming(news)
    input_data = [news]
    vectorized_data = vector_form.transform(input_data)  # Vectorize input data
    prediction = load_model.predict(vectorized_data)  # Pass vectorized data to the model
    return prediction[0]  

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:FINAL@localhost:3306/ai'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

def create_user(username, password):
    user = User(username=username, password=password)
    db.session.add(user)
    db.session.commit()

def authenticate(username, password):
    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        return user
    return None

app.secret_key = "your_secret_key"  

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if "logged_in" in session and session["logged_in"]:
            return render_template("main.html", username=session["username"])

        if username and password:
            user = authenticate(username, password)
            if user:
                session["logged_in"] = True
                session["username"] = user.username
                return render_template("main.html", username=username)
            else:
                return render_template("index.html", error="Invalid username or password")
        else:
            return render_template("index.html", error="Please enter username and password")

    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username and password:
            create_user(username, password)
            return render_template("index.html", message="Registration successful!")
        else:
            return render_template("register.html", error="Please enter username and password")

    return render_template("register.html")

@app.route("/main", methods=["GET", "POST"])
def main():
    if "logged_in" not in session or not session["logged_in"]:
        return render_template("index.html", error="Please login to access this page")

    if request.method == "POST":
        news_content = request.form.get("news_content")
        if news_content:
            preprocessed_text = stemming(news_content)
            prediction = fake_news(preprocessed_text)
            if prediction == 0:
                result = "Reliable"
            else:
                result = "Unreliable"
            return render_template("main.html", username=session["username"], result=result, news_content=news_content)
    return render_template("main.html", username=session["username"], result="", news_content="")  

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    session.pop("username", None)
    return render_template("index.html")

@app.route("/gui")
def gui():
    df = pd.read_csv("train.csv")  # or "test.csv" depending on your data

    # Select 10 random fake news headlines for demonstration
    fake_news_data = df[df["label"] == 1]["text"].sample(10, random_state=42).tolist()

    return render_template("gui.html", fake_news=fake_news_data)


if __name__ == "__main__":
    app.run(debug=True)
