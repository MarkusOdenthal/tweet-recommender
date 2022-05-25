def search_result(
        tweet_id: str,
        tweet: str,
        replied_id: str,
        replied: str,
        score: int,
        like_count: int,
        retweet_count: int,
        author_name: str,
        reply_scores=0.0,
        reply_like_count=0,
        reply_retweet_count=0,
        score_rank=0.0) -> str:
    """ HTML scripts to display search results. """
    return f"""
        <div style="padding: 20px">
            <div style="box-shadow: rgba(0, 0, 0, 0.05) 0px 0px 0px 1px;padding: 20px;background: #d9e6ff; border-radius: 10px;">
                <div style="font-size:140%;">Rank Score: {score_rank} 👤 {author_name}</div>
                <div style="font-size:120%;">
                    <a href="https://twitter.com/i/web/status/{tweet_id}">Tweet:</a> 🟢 {score} | ❤️ {like_count} | 🔄 {retweet_count}
                </div> 
                <div style="font-size:100%;box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;">
                    <p>{tweet}</p>
                </div>
                <div style="font-size:120%;"><a href="https://twitter.com/i/web/status/{replied_id}">Replied:</a> 🟢
                 {reply_scores} | ❤️ {reply_like_count} | 🔄
                 {reply_retweet_count}</div> 
                <div style="font-size:100%;box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;">
                    <p>{replied}</p>
                </div>
            </div>
        </div> 
    """