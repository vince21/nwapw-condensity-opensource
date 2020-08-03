from newsapi import NewsApiClient
from article import Summarizer
import shelve
import time
import os
from tagger import get_tags
from urllib.parse import urlparse
from webscraper import is_valid_url

if __name__ == '__main__':
    while True:
        print("Beginning Scrape...")
        newsapi = NewsApiClient(api_key='1f9393d3ba9d40c3832e87d9088b45cf')
        top_headlines = newsapi.get_top_headlines(language='en',
                                                  country='us')
        articles = []
        for article in top_headlines['articles']:
            if urlparse(article['url']).netloc == 'www.bloomberg.com':
                continue
            try:
                summarizer = Summarizer(article['url'])
                condensed_text = summarizer.condense(100 / len(summarizer.wordlist))
                # test image validity
                if is_valid_url(article['urlToImage']):
                    image = article['urlToImage']
                else:
                    image = None
                articles.append({'Title': article['title'],
                                 'Authors': article['author'],
                                 'Date': article['publishedAt'],
                                 'Text': condensed_text,
                                 'Metrics': summarizer.condense_metrics(condensed_text),
                                 'Image': image,
                                 'Url': article['url'],
                                 'Source': article['source']['name'],
                                 'Tags': get_tags(article['title'], 4)})
            except:
                pass
        news_db = shelve.open('news')
        news_db.clear()
        news_db['data'] = articles
        news_db.close()
        print("Successfully Scraped")
        time.sleep(3600)
