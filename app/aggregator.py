# NWAPW
# Spencer Chang, Toby Ueno, Vincent Wilson
# date: 8/04/20
# description: service to condense articles on hourly basis

from article import Summarizer
import shelve
import time
import os
from tagger import get_tags
from urllib.parse import urlparse
from webscraper import is_valid_url
import requests

"""
This is a service that gets the top articles from News API every hour and puts them in news.db
"""
if __name__ == '__main__':
    while True:
        print("Beginning Scrape...")
        url = ('http://newsapi.org/v2/top-headlines?'
               'country=us&'
               'apiKey=1f9393d3ba9d40c3832e87d9088b45cf')
        response = requests.get(url)

        articles = []
        for article in response.json()['articles']:
            if urlparse(article['url']).netloc == 'www.bloomberg.com':
                continue
            try:
                summarizer = Summarizer(article['url'])
                condensed_text = summarizer.condense(100 / len(summarizer.wordlist))
                # test image validity
                if is_valid_url(article['urlToImage']):
                    image = article['urlToImage']
                else:
                    continue
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
                pass        # throw out any news article with a missing field
        news_db = shelve.open('news')
        news_db.clear()
        news_db['data'] = articles
        news_db.close()
        print("Successfully Scraped")
        time.sleep(3600)
