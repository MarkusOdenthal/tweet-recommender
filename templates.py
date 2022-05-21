def search_result(tweet: str, replied: str, score: int, like_count: int, retweet_count: int) -> str:
    """ HTML scripts to display search results. """
    return f"""
        <div style="box-shadow: rgba(0, 0, 0, 0.05) 0px 0px 0px 1px;;padding: 20px">
            <div style="font-size:140%;">Tweet: 🟢 {score} | ❤️ {like_count} | 🔄 {retweet_count}</div> 
            <div style="font-size:120%;box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;">
                <p>{tweet}</p>
            </div>
            <div>Replied:</div> 
            <div style="font-size:100%;box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;">
                <p>{replied}</p>
            </div>
        </div>
    """