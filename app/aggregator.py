from newsapi import NewsApiClient
from article import Summarizer
import shelve

if __name__ == '__main__':
    newsapi = NewsApiClient(api_key='1f9393d3ba9d40c3832e87d9088b45cf')


    top_headlines = newsapi.get_top_headlines(language='en',
                                              country='us')
    articles = []
    news_db = shelve.open('news.db')
    news_db.clear()

    for article in top_headlines['articles']:
        try:
            summarizer = Summarizer(article['url'])
            articles.append({'Title': article['title'],
                'Authors': article['author'],
                'Date': article['publishedAt'],
                'Text': summarizer.condense(100/len(summarizer.wordlist)),
                'Image': article['urlToImage'],
                'Url': article['url']})
        except:
            pass

    news_db['data'] = articles
    news_db.close()
