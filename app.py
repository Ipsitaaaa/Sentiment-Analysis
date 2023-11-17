from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from email.message import EmailMessage
import ssl
import smtplib
from datetime import datetime
import sqlite3
import torch

load_dotenv()

""" Database Setup """
DB_FILE = "brand_scores_updated.db"
print(f"DB_FILE: {DB_FILE}")



def create_table():
    """ Create a table to store brand scores """
try:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS brand_scores (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      brand_name TEXT UNIQUE,
                      score REAL,
                      review_count INTEGER)''')
    conn.commit()
    conn.close()
    print("Table created successfully.")
except Exception as e:
    print(f"Error creating table: {str(e)}")
    
create_table()

def get_brand_score(brand_name):
    """ Retrieve the score for a given brand from the database """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT score FROM brand_scores WHERE brand_name=?", (brand_name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        print(f"Error in get_brand_score: {str(e)}")
        raise e



def update_brand_score(brand_name, new_score):
    """ Update the score for a given brand in the database """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO brand_scores (brand_name, score, review_count) VALUES (?, ?, ?)",
                    (brand_name, new_score, 0))
        conn.commit()
        conn.close()
        print(f"Brand score updated successfully for {brand_name}")
    except Exception as e:
        print(f"Error updating brand score: {str(e)}")
        raise e

def get_review_count(brand_name):
    """ Retrieve the review count for a given brand from the database """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT review_count FROM brand_scores WHERE brand_name=?", (brand_name,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def update_review_count(brand_name, new_review_count):
    """ Update the review count for a given brand in the database """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE brand_scores SET review_count=? WHERE brand_name=?", (new_review_count, brand_name))
    conn.commit()
    conn.close()


""" Initialization """

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3002",
    "http://localhost:5500"
]

""" Set up CORS for FastAPI """

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

""" Helpers """

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


def map_sentiment_score(review):
    """ Given a review generates the sentiment rating, ranging from 1-5, with 5 being positive """
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1


def worker_function(url):
    """ Accessing Wikipedia to get reviews """
    try:
        r = requests.get(url)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error accessing {url}: {str(e)}")

    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*body-medium-overflow-wrap.*')
    results = soup.find_all('p')
    reviews = [result.text for result in results]

    if not reviews:
        raise HTTPException(status_code=500, detail=f"No reviews found on {url}")

    """ Converting the reviews to a dataframe for easier processing """
    df = pd.DataFrame(np.array(reviews), columns=['reviews'])
    
    df['source'] = 'Reddit' if 'reddit' in url else 'Wikipedia'

    """ Getting the review for each review in our dataframe """
    df['score'] = df['reviews'].apply(lambda x: map_sentiment_score(x))
    print(df)

    """ Returning the final predicted score by taking the mean and scaling it to a 10 point scheme by multiplying 2"""
    # print(f"Final predicted score: {result}")
    return df 


def scrape_function(url):

    """ Accesing the web to get reviews """

    try:
        r = requests.get(url)
        r.raise_for_status()  
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error accessing {url}: {str(e)}")

    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*_1poyrkZ7g36PawDueRza-J _11R7M_VOgKO1RJyRSRErT3 _1Qs6zz6oqdrQbR7yE_ntfY.*')
       
    # results = soup.find_all('p', {'class':regex})
    results = soup.find_all('p')
    reviews = [result.text for result in results]

    if not reviews:
        raise HTTPException(status_code=500, detail=f"No reviews found on ")


    """ Converting the reviews to a dataframe for easier processing """

    df = pd.DataFrame(np.array(reviews), columns=['reviews'])

    df['source'] = 'Reddit' if 'reddit' in url else 'Wikipedia'

    df['score'] = df['reviews'].apply(lambda x: map_sentiment_score(x))

    print(df)

    return df

"""  Mail Setup """

sender_email = 'rishi.gnit2025@gmail.com'
sender_password = os.getenv('EMAIL_PASSWORD')


def broadcast(keyword, ans):
    email_list = [
        'ipsitanayak09765@gmail.com',
        'rk04011@outlook.com',
        'dummy@gmail.com'
    ]

    for email in email_list:
        send_mail(email, keyword, ans)


def send_mail(receiver_email, keyword, data=None):
    date = datetime.now().strftime("Date: %d/%m/%Y Time: %H:%M:%S")
    subject = "Check your result for the sentiment analysis!"
    body = f"The sentiment score for {keyword.title()} is {data:.2f}. Review generated on {date}"

    em = EmailMessage()
    em['From'] = sender_email
    em['To'] = receiver_email
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.sendmail(sender_email, receiver_email, em.as_string())


""" Endpoints """

@app.get("/")
async def root():
    return {"Server is running successfully!"}


@app.post("/get_score")
async def get_score(request: Request):
    try:
        req_body = await request.json()
        if not req_body:
            raise HTTPException(status_code=400, detail="Empty JSON body")

        keyword = req_body.get("keyword").lower()

        if not keyword:
            raise HTTPException(status_code=400, detail="Missing 'keyword' in the request body")

        print(f"Processing keyword: {keyword}")

        wiki_url = f"https://en.wikipedia.org/wiki/{keyword}"
        reddit_url = f"https://www.reddit.com/r/{keyword}/"

        wiki_df = worker_function(wiki_url)  
        reddit_df = scrape_function(reddit_url)

        combined_df = pd.concat([wiki_df, reddit_df], ignore_index=True)

        # Perform sentiment analysis or any other operations on the combined data
        combined_df['score'] = combined_df['reviews'].apply(lambda x: map_sentiment_score(x))
        ans = combined_df['score'].mean() * 2

        print(f"Retrieving existing score and review count for {keyword}")
        # Retrieve existing score and number of reviews from the database
        existing_score = get_brand_score(keyword)
        existing_review_count = get_review_count(keyword)

        print(f"Existing score: {existing_score}, Existing review count: {existing_review_count}")

        if existing_score is None:
            print(f"No existing score found. Setting initial score to {ans}")
            existing_score = ans
            existing_review_count = 1
            update_brand_score(keyword, existing_score)
            update_review_count(keyword, existing_review_count)
            broadcast(keyword, existing_score)
            return {'data': "{:.2f}".format(existing_score)}

        print(f"Calculating modified score based on sentiment analysis result: {ans}")
        if isinstance(ans, float):
            # If ans is a single float value, consider it as one review
            existing_review_count += 1
        else:
            # If ans is a list of scores, update the review count accordingly
            existing_review_count += len(ans)


        # Calculate the modified score based on the ratio of positive reviews
        modified_score = existing_score + 1 if (existing_score / existing_review_count < 5) else existing_score - 1


        print(f"Modified score: {modified_score}")

        # Update the score and review count in the database
        update_brand_score(keyword, modified_score)
        update_review_count(keyword, existing_review_count)

        # Broadcast the modified score via email
        broadcast(keyword, modified_score)

        return {'data': "{:.2f}".format(modified_score)}
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")