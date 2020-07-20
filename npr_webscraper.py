import requests
from bs4 import BeautifulSoup


def scrape(url):
    """
    Takes a URL for an NPR article and returns the text and images.
    :param url: NPR article link
    :type url: str
    :return: List of text elements in body and list of image links
    :rtype: ([bs4.element.Tag], [str]))
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

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

    return text_elements, images


def get_text(tags):
    """
    Converts a list of tags into a string.
    :param tags: List of tags to be converted
    :type tags: [bs4.element.Tag]
    :return: String containing the text from within each tag
    :rtype: str
    """
    return '\n'.join([tag.text for tag in tags])


if __name__ == '__main__':
    test_url = 'https://www.npr.org/2020/07/20/891854646/whales-get-a-break-as-pandemic-creates-quieter-oceans'
    print(get_text(scrape(test_url)[0]))
    print(scrape(test_url)[1])
