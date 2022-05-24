import streamlit as st
from supabase import create_client
import httpx
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from templates import search_result

st.set_page_config(layout="wide")


# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    return create_client(url, key)


supabase = init_connection()
embedder = SentenceTransformer('all-MiniLM-L6-v2')


# Perform query.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=3600)
def run_query(username: str):
    author_id = supabase.table("twitter_author_id_to_author_username"). \
        select("*").eq('author_username', username). \
        execute().\
        data[0]['author_id']
    response = supabase.table("tweets").select("*").eq('from_author_id', author_id).execute()

    df = pd.DataFrame(response.data)

    df_tweet = df[df["referenced_tweets_type"] == "tweet"].copy()
    df_tweet = df_tweet[~df_tweet['tweet_id'].duplicated()]
    corpus = df_tweet['text'].tolist()
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    return df, df_tweet, corpus, corpus_embeddings


def main():
    get_twitter_timeline_hook = st.secrets["get_twitter_timeline_hook"]
    username = st.text_input('Enter Twitter Username:')

    if username:
        r = httpx.post(get_twitter_timeline_hook, data={'username': username})
        df, df_tweet, corpus, corpus_embeddings = run_query(username)

    search = st.text_input('Enter search tweet:')
    if search:
        query_embedding = embedder.encode(search, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=10)

        # Print results.

        for score, idx in zip(top_results[0], top_results[1]):
            tweet = df_tweet.iloc[int(idx)]
            tweet_id = tweet['tweet_id']
            tweet_like_count = tweet['like_count']
            retweet_count = tweet['retweet_count']
            reply_tweet = df[df['referenced_tweets_id'] == tweet_id]
            if not reply_tweet.empty:
                df_repl = reply_tweet['text'].iloc[0]
            else:
                df_repl = ""

            st.write(search_result(f'{corpus[idx]}', replied=f"{df_repl}", score="Score: {:.4f}".format(score),
                                   like_count=tweet_like_count, retweet_count=retweet_count), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
