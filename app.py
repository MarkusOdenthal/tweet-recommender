"""
Main module and starting point for streamlit of project tweet recommender.
"""

import collections
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from supabase import create_client
import httpx
import pandas as pd
import torch
from methods.templates import search_result


# set page title and icon
st.set_page_config(
    page_title="Tweet Engage Recommender",
    page_icon="🐦",
    layout="wide"
)

# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    return create_client(url, key)


supabase = init_connection()
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Perform query.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=3600)
def run_query(author_id: str):
    response = (
        supabase.table("tweets").select("*").eq("from_author_id", author_id).execute()
    )

    df = pd.DataFrame(response.data)

    df_tweet = df[df["referenced_tweets_type"] == "tweet"].copy()
    df_tweet = df_tweet[~df_tweet["tweet_id"].duplicated()]
    corpus = df_tweet["text"].tolist()
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    return df, df_tweet, corpus, corpus_embeddings


def query_tweet_metric(author_id: str) -> int:
    response = (
        supabase.table("tweets")
            .select("referenced_tweet_fetch")
            .eq("from_author_id", author_id)
            .execute()
    )
    metric_data = pd.DataFrame(response.data)
    metrics = metric_data['referenced_tweet_fetch'].value_counts()
    true_count_metric = metrics.get(True, default=0)
    false_count_metric = metrics.get(False, default=0)
    tweet_fetch = (1-false_count_metric/(false_count_metric+true_count_metric))*100
    tweet_fetch = int(tweet_fetch)
    return tweet_fetch


def get_twitter_author_id(username: str) -> str:
    try:
        author_id = (
            supabase.table("twitter_author_id_to_author_username")
            .select("*")
            .eq("author_username", username)
            .execute()
            .data[0]["author_id"]
        )
    except:
        get_twitter_user_lookup = st.secrets["get_twitter_user_lookup"]
        r = httpx.post(get_twitter_user_lookup, data={"username": username})
        author_id = (
            supabase.table("twitter_author_id_to_author_username")
            .select("*")
            .eq("author_username", username)
            .execute()
            .data[0]["author_id"]
        )
    return author_id


st.markdown("###### Made with :heart: by @MarkusOdenthal | [![Follow]("
                "https://img.shields.io/twitter/follow/MarkusOdenthal?style=social)](https://www.twitter.com/MarkusOdenthal) | [![this is an image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/markusodenA)")

with st.expander("ℹ️ About this app", expanded=True):

    st.write(
        """     
-   The *Tweet Engage Recommender MVP* app is an easy-to-use interface built in Streamlit it should give you an idea for the future product
-   It can be tough to keep up with conversation on Twitter, but Tweet Recommender is here to help you out
-   It's the Semantic Search Engine for your past tweets, making it easy to find similar tweets and join in the discussion. Never be out of the loop again!
-   The tool is still in Beta. Any issues, feedback or suggestions please DM me on Twitter: [@MarkusOdenthal](https://twitter.com/MarkusOdenthal)
-   This app is free. If it's useful to you, you can [buy me a ☕](https://www.buymeacoffee.com/markusodenA) to support my work. 🙏
	    """
    )

    st.markdown("")

with st.expander("🔆 Coming soon!", expanded=False):

    st.write(
        """  
-   Add more embedding models.
-   Add more options from the KeyBERT API.
-   Allow for larger documents to be reviewed (currently limited to 500 words).
-   If deemed cost-effective, I may add search volume data with each retrieved keyword.
	    """
    )

    st.markdown("")

st.markdown("")

st.markdown("## 🔍 Search Tweets")


def main():
    with st.form(key="my_form"):
        ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
        with c1:
            username = st.text_input("Enter Twitter Username:", help="Please enter here you Twitter Username without the @")

            if username:
                author_id = get_twitter_author_id(username)
                value = query_tweet_metric(author_id)
            else:
                value = 0

            label = "Tweets fetch %"
            st.metric(label, value, delta=None)

            replied_weighting = st.slider(
                "Replied weighting %",
                min_value=0,
                max_value=30,
                value=6,
                help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
            )
        with c2:
            get_twitter_timeline_hook = st.secrets["get_twitter_timeline_hook"]

            if username:
                _ = httpx.post(get_twitter_timeline_hook, data={"username": username})
                df, df_tweet, corpus, corpus_embeddings = run_query(author_id)

            search = st.text_input("Enter search tweet:")
            st.form_submit_button('Search')
            if search:
                query_embedding = embedder.encode(search, convert_to_tensor=True)

                cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                top_results = torch.topk(cos_scores, k=10)

                # Print results.
                hits = {}
                tweet_score_weights = 1 - replied_weighting/100
                replied_score_weights = replied_weighting/100
                for score, idx in zip(top_results[0], top_results[1]):
                    hit = {}
                    tweet = df_tweet.iloc[int(idx)]
                    tweet_id = tweet["tweet_id"]
                    author_name = tweet["author_name"]
                    tweet_like_count = tweet["like_count"]
                    tweet_retweet_count = tweet["retweet_count"]
                    reply_tweet = df[df["referenced_tweets_id"] == tweet_id]
                    if not reply_tweet.empty:
                        first_reply = reply_tweet.iloc[0]
                        text_repl = first_reply["text"]
                        replied_id = first_reply["tweet_id"]
                        text_repl = " ".join(filter(lambda x: x[0] != "@", text_repl.split()))
                        corpus_reply_embeddings = embedder.encode(
                            text_repl, convert_to_tensor=True
                        )
                        reply_scores = float(
                            util.cos_sim(query_embedding, corpus_reply_embeddings)[0]
                        )
                        reply_like_count = first_reply["like_count"]
                        reply_retweet_count = first_reply["retweet_count"]
                    else:
                        text_repl = ""

                    score_rank = tweet_score_weights * score + replied_score_weights * reply_scores
                    score_rank = float(score_rank)

                    hit["tweet_id"] = tweet_id
                    hit["tweet"] = corpus[idx]
                    hit["tweet_score"] = "Score: {:.2f}".format(score)
                    hit["author_name"] = author_name
                    hit["tweet_like_count"] = tweet_like_count
                    hit["tweet_retweet_count"] = tweet_retweet_count

                    hit["replied_id"] = replied_id
                    hit["text_repl"] = text_repl
                    hit["reply_score"] = "Score: {:.2f}".format(reply_scores)
                    hit["reply_like_count"] = reply_like_count
                    hit["reply_retweet_count"] = reply_retweet_count
                    hit["score_rank"] = "{:.2f}".format(score_rank)

                    hits[score_rank] = hit

                # do reranking of the results
                rerank_hits = collections.OrderedDict(sorted(hits.items(), reverse=True))
                for rerank_score, hit in rerank_hits.items():
                    st.write(
                        search_result(
                            f"{hit['tweet_id']}",
                            f"{hit['tweet']}",
                            replied_id=f"{hit['replied_id']}",
                            replied=f"{hit['text_repl']}",
                            score=hit["tweet_score"],
                            like_count=hit["tweet_like_count"],
                            retweet_count=hit["tweet_retweet_count"],
                            reply_scores=hit["reply_score"],
                            reply_like_count=hit["reply_like_count"],
                            reply_retweet_count=hit["reply_retweet_count"],
                            score_rank=hit["score_rank"],
                            author_name=hit["author_name"],
                        ),
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    query_tweet_metric('753027473193873408')
    # get_twitter_author_id('test')
    # run_query('MarkusOdenthal')
    main()
