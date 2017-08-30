
# coding: utf-8

# In[36]:

#
# https://quizlet.com/7351070/flashcards

import numpy
from selenium import webdriver
import time
from time import perf_counter

from bs4 import BeautifulSoup

from pyvirtualdisplay import Display

PHANTOMJS_PATH = '/usr/bin/phantomjs'

NOTEBOOK = False


# In[123]:

def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def get_flashcards_text(page_url,
                        driver=None,
                        display=None,
                        card_content_id='CardsItem',
                        phantomjs_path=PHANTOMJS_PATH,
                        waiting_time=20,
                        close=False):

    if not display:
        display = Display(visible=0, size=(800, 600))
        display.start()

    #
    # initing phantomjs driver
    # TODO: I have to turn this into a context manager...
    if not driver:
        # driver = webdriver.PhantomJS(executable_path=phantomjs_path, service_args=['--ignore-ssl-errors=true'])
        driver = webdriver.Firefox()

    card_texts = []
    print('Processing page', page_url, end='     \r')
    get_start_t = perf_counter()
    #
    # making the request
    driver.get(page_url)
    get_end_t = perf_counter()
    print('\t\t\tgot request in {} secs'.format(get_end_t - get_start_t), end='       \r')

    proc_start_t = perf_counter()

    #
    # waiting some time to let it render
    time.sleep(waiting_time)

    #
    # getting all cards
    # print(driver.page_source)
    html_source = driver.page_source
    # print(html_source)
    # print(html_source)
    if len(html_source) > 500:
        soup = BeautifulSoup(html_source, 'lxml')
        section = soup.find("section", {"class": "CardsSet"})
        for f in section.select('span.TermText'):
            card_texts.append(f.text)

    else:
        raise RuntimeError('short Html {}'.format(html_source))

    qas = list(pairwise(card_texts))

    proc_end_t = perf_counter()
    print('Processed {0} cards in {1}'.format(
        len(qas), proc_end_t - proc_start_t), end='       \r')

    #
    # closing
    if close:
        driver.close()
        display.stop()

    return qas


# In[124]:

def save_qa_list_to_file(file_path,
                         qa_list,
                         separator='\t'):
    with open(file_path, 'w') as qa_file:
        for q, a in qa_list:
            qa_line = q + separator + a + '\n'
            qa_file.write(qa_line)


# In[125]:

if NOTEBOOK:
    url_ex = 'http://quizlet.com/7351070/flashcards'
    qas = get_flashcards_text(url_ex)
    for q, a in qas:
        print('Q:', q)
        print('A:', a)
    save_qa_list_to_file('qa-test.tsv', qas)


# In[120]:

BASE_URL = 'http://quizlet.com/'


def compose_url_from_id(id):
    return BASE_URL + '{}/flashcards'.format(id)


def scrape_flashcard_url_list(page_url,
                              driver=None,
                              card_content_id='currentCard',
                              phantomjs_path=PHANTOMJS_PATH,
                              waiting_time=20,
                              close=False):

    #
    # initing phantomjs driver
    # TODO: I have to turn this into a context manager...
    if not driver:
        # driver = webdriver.PhantomJS(executable_path=phantomjs_path)
        driver = webdriver.Firefox()

    get_start_t = perf_counter()
    #
    # making the request
    print('url request', page_url)

    driver.get(page_url)
    get_end_t = perf_counter()
    print('\t\t\tgot request in {} secs'.format(get_end_t - get_start_t), end='       \r')

    time.sleep(waiting_time)

    #
    # retrieving hrefs
    flashcards_links = driver.find_elements_by_xpath("//div[contains(@class, 'SearchResult ')]")
    flashcards_links = [
        compose_url_from_id(a.get_attribute('data-item-id')) for a in flashcards_links]

    #
    # closing
    # if close:
    driver.close()

    return flashcards_links


# In[122]:

# https://quizlet.com/subject/science/page/2/
if NOTEBOOK:
    url_ex = 'https://quizlet.com/subject/science/page/1/'
    fls = scrape_flashcard_url_list(url_ex)
    for f in fls:
        print(f)


# In[126]:

from collections import deque
import os

EXT = '.tsv'


def scrape_flashcards_from_page(page_url,
                                page_id,
                                display=None,
                                output_dir='../data/quizlet.com/flashcards/Astronomy/',
                                card_content_id='currentCard',
                                phantomjs_path=PHANTOMJS_PATH,
                                waiting_time=2):

    if display is None:
        display = Display(visible=0, size=(800, 600))
        display.start()
    #
    # retrieve the url for each flashcard from the page
    url_list = scrape_flashcard_url_list(page_url)
    print('There are {} decks'.format(len(url_list)))

    #
    # saving them in an index
    output_dir = output_dir + '{0}/'.format(page_id)
    os.makedirs(output_dir, exist_ok=True)
    index_path = output_dir + '{0}.index'.format(page_id)
    with open(index_path, 'w') as index_file:
        for url in url_list:
            index_file.write(url + '\n')

    urls_to_process = deque(url_list)
    n_urls = len(url_list)
    i = 0

    #
    # instantiating a single driver
    # driver = webdriver.PhantomJS(executable_path=phantomjs_path)
    driver = webdriver.Firefox()
    driver.set_page_load_timeout(120)

    page_start_t = perf_counter()

    while urls_to_process:

        try:
            #
            # take the first one
            flashcard_url = urls_to_process.popleft()
            flashcard_id = flashcard_url.split('/')[-2]

            if flashcard_id == '64784199':
                continue
            print('- [{0}/{1}@{3}] Processing flashcard {2}'.format(i + 1,
                                                                    n_urls,
                                                                    flashcard_id,
                                                                    page_id))

            #
            # get path and check that it has not been processed yet
            output_path = output_dir + flashcard_id + EXT
            if not os.path.exists(output_path):

                #
                # get qa
                q_start_t = perf_counter()
                qas = get_flashcards_text(flashcard_url, driver=driver, display=display)
                q_end_t = perf_counter()
                print('flashcard extracted in {}'.format(q_end_t - q_start_t))

                #
                # saving
                save_qa_list_to_file(output_path, qas)
                print('\tsaved to file {}'.format(output_path), end='            \r')

            else:
                print('File already exists, skipping')
            i += 1

        except Exception as e:

            print('\t**** [{0}] Error while processing {1}, retrying ****'.format(e,
                                                                                  flashcard_url))
            urls_to_process.appendleft(flashcard_url)
            driver.close()
            driver = webdriver.Firefox()
            driver.set_page_load_timeout(120)

    page_end_t = perf_counter()
    print('\n!!!! page completed in {} secs !!!!'.format(page_end_t - page_start_t))

    driver.close()

    if display:
        display.stop()


if NOTEBOOK:
    url = 'http://quizlet.com/subject/science/page/1/'
    output_path = '../data/quizlet.com/flashcards/'
    scrape_flashcards_from_page(url, 1, output_path)


def generate_url_pages_list(base_url, min_page_num, max_page_num):
    return [base_url + str(i) + '/' for i in range(min_page_num, max_page_num)]

import argparse

BASE_URL_PAGE = 'http://quizlet.com/subject/astronomy/page/'
OUT_PATH = '../data/quizlet.com/flashcards/Astronomy/'
N_PROCESSES = 1
MIN_PAG = 1
MAX_PAG = 10

if __name__ == '__main__':
    #
    # creating the parser for input args
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str,
                        default=OUT_PATH,
                        help='output directory')

    parser.add_argument('-f', '--first-page', type=int,
                        default=MIN_PAG,
                        help='number of page to start from')

    parser.add_argument('-l', '--last-page', type=int,
                        default=MAX_PAG,
                        help='number of page to stop to')

    parser.add_argument('-u', '--base-url', type=str,
                        default=BASE_URL_PAGE,
                        help='base url')

    # parsing the args
    args = parser.parse_args()

    #
    # generating page urls
    pages_urls = generate_url_pages_list(args.base_url, args.first_page, args.last_page + 1)

    #
    # starting scraping
    for i, url in zip(range(args.first_page, args.last_page + 1), pages_urls):
        print('\n\n\nProcessing page: {}\n\n\n'.format(url))
        scrape_flashcards_from_page(url, i, output_dir=args.output)
