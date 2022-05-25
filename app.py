import streamlit as st
from supabase import create_client
import httpx
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from methods.templates import search_result
import collections

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
        hits = {}
        for score, idx in zip(top_results[0], top_results[1]):
            hit = {}
            tweet = df_tweet.iloc[int(idx)]
            tweet_id = tweet['tweet_id']
            author_name = tweet['author_name']
            tweet_like_count = tweet['like_count']
            tweet_retweet_count = tweet['retweet_count']
            reply_tweet = df[df['referenced_tweets_id'] == tweet_id]
            if not reply_tweet.empty:
                first_reply = reply_tweet.iloc[0]
                text_repl = first_reply['text']
                replied_id = first_reply['tweet_id']
                text_repl = " ".join(filter(lambda x: x[0] != '@', text_repl.split()))
                corpus_reply_embeddings = embedder.encode(text_repl, convert_to_tensor=True)
                reply_scores = float(util.cos_sim(query_embedding, corpus_reply_embeddings)[0])
                reply_like_count = first_reply['like_count']
                reply_retweet_count = first_reply['retweet_count']
            else:
                text_repl = ""
            score_rank = 0.7 * score + 0.3 * reply_scores
            score_rank = float(score_rank)

            hit['tweet_id'] = tweet_id
            hit['tweet'] = corpus[idx]
            hit['tweet_score'] = "Score: {:.2f}".format(score)
            hit['author_name'] = author_name
            hit['tweet_like_count'] = tweet_like_count
            hit['tweet_retweet_count'] = tweet_retweet_count

            hit['replied_id'] = replied_id
            hit['text_repl'] = text_repl
            hit['reply_score'] = "Score: {:.2f}".format(reply_scores)
            hit['reply_like_count'] = reply_like_count
            hit['reply_retweet_count'] = reply_retweet_count
            hit['score_rank'] = "{:.2f}".format(score_rank)

            hits[score_rank] = hit

        # do reranking of the results
        rerank_hits = collections.OrderedDict(sorted(hits.items(), reverse=True))
        for rerank_score, hit in rerank_hits.items():
            st.write(search_result(f"{hit['tweet_id']}", f"{hit['tweet']}", replied_id=f"{hit['replied_id']}",
                                   replied=f"{hit['text_repl']}",
                                   score=hit['tweet_score'],
                                   like_count=hit['tweet_like_count'], retweet_count=hit['tweet_retweet_count'],
                                   reply_scores=hit['reply_score'],
                                   reply_like_count=hit['reply_like_count'], reply_retweet_count=hit['reply_retweet_count'],
                                   score_rank=hit['score_rank'], author_name=hit['author_name']),
                     unsafe_allow_html=True)


if __name__ == '__main__':
    main()
