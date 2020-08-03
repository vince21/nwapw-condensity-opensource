from newspaper import Article
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError
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

    output_dict = {'Title': title,
                   'Authors': authors,
                   'Text': raw_text
                   }

    return output_dict


def bbc_scrape(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    try:
        title = soup.find('h1', class_='story-body__h1').text.strip()
    except AttributeError:
        return None

    try:
        authors = [soup.find('span', class_='byline__name').text.strip()[3:]]
    except AttributeError:
        authors = []

    body_text = soup.find('div', class_='story-body__inner').find_all('p', recursive=False)
    # removes ad manually (only appears on some articles)
    if body_text[-1].text[:21] == 'Follow us on Facebook':
        body_text = body_text[:-1]
    raw_text = '\n'.join([tag.text.strip() for tag in body_text])

    output_dict = {'Title': title,
                   'Authors': authors,
                   'Text': raw_text
                   }

    return output_dict


def atlantic_scrape(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    try:
        title = soup.find('h1', class_='c-article-header__hed').text.strip()
    except AttributeError:
        return None

    authors = soup.find_all('span', class_='c-byline__author')
    authors = [author.text.strip() for author in authors]

    body_text = soup.find_all('p', {'dir': 'ltr', 'data-id': False})
    raw_text = '\n'.join([tag.text.strip() for tag in body_text])

    output_dict = {'Title': title,
                   'Authors': authors,
                   'Text': raw_text}

    return output_dict


def is_valid_url(url):
    valid_syntax = False
    try:
        parsed_url = urlparse(url)
        valid_syntax = all([parsed_url.scheme, parsed_url.netloc])
    except ValueError:
        valid_syntax = False
    if valid_syntax:
        try:
            return urlopen(url).getcode() == 200
        except URLError:
            return False


def default_scrape(url, data={}):
    article = Article(url)
    article.download()
    article.parse()
    article_info = {}
    for k, v in data.items():
        if k in ['Title', 'Authors', 'Date', 'Text', 'Image']:
            article_info[k] = v
    article_info['Title'] = article_info.get('Title', article.title)
    article_info['Authors'] = article_info.get('Authors', article.authors)
    article_info['Date'] = article_info.get('Date', article.publish_date)
    article_info['Text'] = article_info.get('Text', article.text)
    article_info['Image'] = article_info.get('Image', article.top_image)

    return article_info


def scrape(url):
    """
    Takes a url and returns info about the page.
    :param url: Valid url for a web article
    :type url: str
    :return: Article title, list of authors, date published (datetime), text, and an image
    :rtype: dict
    """

    if not is_valid_url(url):
        return None

    domain = urlparse(url).netloc.split('.')[1]

    if domain == 'npr':
        output_dict = npr_scrape(url)
    elif domain == 'washingtonpost':
        output_dict = wapo_scrape(url)
    elif domain == 'bbc':
        output_dict = bbc_scrape(url)
    elif domain == 'theatlantic':
        output_dict = atlantic_scrape(url)

    if not output_dict:
        output_dict = {}

    output_dict = default_scrape(url, output_dict)

    return output_dict


if __name__ == '__main__':
    test_url = 'https://www.theatlantic.com/magazine/archive/2020/09/coronavirus-american-failure/614191/'
    scrape_output = scrape(test_url)
    print(f'Title: {scrape_output["Title"]}')
    print(f'Authors: {scrape_output["Authors"]}')
    print(f'Date: {scrape_output["Date"]}')
    print(f'Text: {scrape_output["Text"]}')
    print(f'Image: {scrape_output["Image"]}')
