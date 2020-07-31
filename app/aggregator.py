from newsapi import NewsApiClient
from article import Summarizer
import shelve
import time
import os
from tagger import get_tags

if __name__ == '__main__':
    while True:
        print("Beginning Scrape...")
        newsapi = NewsApiClient(api_key='1f9393d3ba9d40c3832e87d9088b45cf')
        top_headlines = newsapi.get_top_headlines(language='en',
                                                  country='us')
        articles = []
        os.remove('news')
        news_db = shelve.open('news')
        # news_db.clear()
        for article in top_headlines['articles']:
            try:
                summarizer = Summarizer(article['url'])
                condensed_text = summarizer.condense(100/len(summarizer.wordlist))
                articles.append({'Title': article['title'],
                    'Authors': article['author'],
                    'Date': article['publishedAt'],
                    'Text': condensed_text,
                    'Metrics': summarizer.condense_metrics(condensed_text),
                    'Image': article['urlToImage'],
                    'Url': article['url'],
                    'Source': article['source']['name'],
                    'Tags': get_tags(article['title'], 4)})
            except:
                pass

        news_db['data'] = articles
        news_db.close()
        print("Successfully Scraped")
        time.sleep(3600)
