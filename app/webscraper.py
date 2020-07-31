from newspaper import Article
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from datetime import datetime


def npr_scrape(url, write=False):
    """
    Takes a URL for an NPR article and returns the page text and info.
    :param url: NPR article link
    :type url: str
    :param write: If true, writes the article text to a file
    :type write: bool
    :return: Article title, list of authors, date published( datetime), text, and an image
    :rtype: dict
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    try:
        title = soup.find('div', class_='storytitle').find('h1').text.strip()
    except AttributeError:
        return None

    author = soup.find('p', class_='byline__name byline__name--block').text.strip()

    date_text = soup.find('div', class_='dateblock').find('time')['datetime']
    date = datetime.strptime(date_text, '%Y-%m-%dT%X%z')

    text_elements = []
    images = []

    body = soup.find('div', id='storytext')
    for element in body.children:
        if element.name == 'p':
            text_elements.append(element)
        elif element.name == 'div':
            element_images = element.find_all('img')
            element_images = [img['data-original'] for img in element_images]
            images += element_images

    raw_text = '\n'.join([tag.text.strip() for tag in text_elements])

    output_dict = {'Title': title,
                   'Authors': [author],
                   'Date': date,
                   'Text': raw_text,
                   'Image': images[0]
                   }

    if write:
        modified_title = '-'.join(title.lower().split(' '))
        with open(f'scraped-text/{modified_title}.txt', 'w') as f:
            f.write(raw_text)

    return output_dict


def wapo_scrape(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    try:
        title = soup.find('h1', class_='font--headline').text.strip()
    except AttributeError:
        return None

    authors = soup.find_all('span', class_='author-name')
    authors = [author.text.strip() for author in authors]

    body_text = soup.find_all('p', class_='font--body')
    raw_text = '\n'.join([tag.parent.text.strip() for tag in body_text])

    article = Article(url)
    article.download()
    article.parse()

    output_dict = {'Title': title,
                   'Authors': authors,
                   'Date': article.publish_date,
                   'Text': raw_text,
                   'Image': article.top_image
                   }

    return output_dict


def scrape(url):
    """
    Takes a url and returns info about the page.
    :param url: Valid url for a web article
    :type url: str
    :return: Article title, list of authors, date published( datetime), text, and an image
    :rtype: dict
    """

    domain = urlparse(url).netloc.split('.')[1]
    if domain == 'npr':
        output_dict = npr_scrape(url)
    elif domain == 'washingtonpost':
        output_dict = wapo_scrape(url)
    else:
        article = Article(url)
        try:
            article.download()
            article.parse()
            output_dict = {'Title': article.title,
                           'Authors': article.authors,
                           'Date': article.publish_date,
                           'Text': article.text,
                           'Image': article.top_image}
        except:
            output_dict = None

    if output_dict is None:
        output_dict = {'Title': None,
                       'Authors': None,
                       'Date': None,
                       'Text': None,
                       'Image': None}

    return output_dict


if __name__ == '__main__':
    test_url = 'https://www.npr.org/'
    scrape_output = scrape(test_url)
    print(f'Title: {scrape_output["Title"]}')
    print(f'Authors: {scrape_output["Authors"]}')
    print(f'Date: {scrape_output["Date"]}')
    print(f'Text: {scrape_output["Text"]}')
    print(f'Image: {scrape_output["Image"]}')
