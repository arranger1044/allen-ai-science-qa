
# coding: utf-8

# In[1]:

#
# http://www.studystack.com/flashcard-1169009
import numpy
from selenium import webdriver
import time
from time import perf_counter


PHANTOMJS_PATH = '/usr/bin/phantomjs'
NOTEBOOK = False


# In[2]:

def get_flashcards_text(page_url,
                        driver=None,
                        card_content_id='currentCard',
                        phantomjs_path=PHANTOMJS_PATH,
                        waiting_time=2,
                        click_waiting_time=0.5,
                        close=False):

    #
    # initing phantomjs driver
    # TODO: I have to turn this into a context manager...
    if not driver:
        driver = webdriver.PhantomJS(executable_path=phantomjs_path)

    qas = []
    get_start_t = perf_counter()
    #
    # making the request
    driver.get(page_url)
    get_end_t = perf_counter()
    print('\t\t\tgot request in {} secs'.format(get_end_t - get_start_t), end='       \r')

    #
    # waiting some time to let it render
    time.sleep(waiting_time)

    #
    # check remaining cards
    # remaining_box = driver.find_element_by_id('remainingBox')
    flashcard = driver.find_element_by_id(card_content_id)
    while flashcard.is_displayed():

        remaining_box = driver.find_element_by_id('remainingBox')
        print('\tRemaining cards', remaining_box.text, end='      \r')
        extract_start_t = perf_counter()
        #
        # retrieve the content of the flashcard question

        flashcard_table = flashcard.find_element_by_tag_name('table')
        flashcard_question = flashcard_table.text
        # print('Q:', flashcard_question)
        #
        # simulate a click to get the answer
        flashcard.click()
        time.sleep(click_waiting_time)
        flashcard_table = flashcard.find_element_by_tag_name('table')
        flashcard_answer = flashcard_table.text
        # print('A:', flashcard_answer)

        extract_end_t = perf_counter()
        print('\t\t\textracted content in {} secs'.format(
            extract_end_t - extract_start_t),  end='       \r')

        #
        # click on don't know to move to next card
        incorrect_box = driver.find_element_by_id('incorrectBox')
        incorrect_box.click()

        #
        # check again for remaining
        time.sleep(click_waiting_time)
        remaining_box = driver.find_element_by_id('remainingBox')

        qas.append((flashcard_question, flashcard_answer))
        flashcard = driver.find_element_by_id(card_content_id)

    #
    # closing
    if close:
        driver.close()

    return qas


# In[3]:

def save_qa_list_to_file(file_path,
                         qa_list,
                         separator='\t'):
    with open(file_path, 'w') as qa_file:
        for q, a in qa_list:
            qa_line = q + separator + a + '\n'
            qa_file.write(qa_line)


# In[5]:

if NOTEBOOK:
    url_ex = 'http://www.studystack.com/flashcard-1169009'
    qas = get_flashcards_text(url_ex)
    for q, a in qas:
        print('Q:', q)
        print('A:', a)
    save_qa_list_to_file('qa-test.tsv', qas)


# In[4]:

def scrape_flashcard_url_list(page_url,
                              driver=None,
                              card_content_id='currentCard',
                              phantomjs_path=PHANTOMJS_PATH,
                              waiting_time=2,
                              close=False):

    #
    # initing phantomjs driver
    # TODO: I have to turn this into a context manager...
    if not driver:
        driver = webdriver.PhantomJS(executable_path=phantomjs_path)

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
    flashcards_links = driver.find_elements_by_xpath(
        "//div[contains(@class, 'stackItem')]/span[@class='description']/a")
    flashcards_links = [a.get_attribute('href') for a in flashcards_links]

    #
    # closing
    if close:
        driver.close()

    return flashcards_links


# In[58]:

if NOTEBOOK:
    url_ex = 'http://www.studystack.com/Science'
    fls = scrape_flashcard_url_list(url_ex)
    for f in fls:
        print(f)


# In[5]:

from collections import deque
import os

EXT = '.tsv'


def scrape_flashcards_from_page(page_url,
                                page_id,
                                output_dir='../data/studystack.com/flashcards/Physics/',
                                card_content_id='currentCard',
                                phantomjs_path=PHANTOMJS_PATH,
                                waiting_time=2):

    #
    # retrieve the url for each flashcard from the page
    url_list = scrape_flashcard_url_list(page_url)

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
    print('There are {} flashcard decks'.format(n_urls))
    i = 0

    #
    # instantiating a single driver
    driver = webdriver.PhantomJS(executable_path=phantomjs_path)
    # driver = webdriver.Firefox()
    driver.set_page_load_timeout(40)

    page_start_t = perf_counter()

    while urls_to_process:

        try:
            #
            # take the first one
            flashcard_url = urls_to_process.popleft()
            flashcard_id = flashcard_url.split('/')[-1]
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
                qas = get_flashcards_text(flashcard_url, driver=driver)
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
            driver = webdriver.PhantomJS(executable_path=phantomjs_path)
            # driver = webdriver.Firefox()
            driver.set_page_load_timeout(40)
            urls_to_process.appendleft(flashcard_url)

    page_end_t = perf_counter()
    print('\n!!!! page completed in {} secs !!!!'.format(page_end_t - page_start_t))

    driver.close()

    # return flashcards_links


# In[ ]:

if NOTEBOOK:
    url = 'http://www.studystack.com/Science&sortOrder=stars&page=1'
    output_path = '../data/studystack.com/flashcards/'
    scrape_flashcards_from_page(url, 1, output_path)


# In[ ]:

from multiprocessing import Pool


def scrape_flashcards_helper(args):
    return scrape_flashcards_from_page(*args)


def scrape_multiple_pages(page_urls_list,
                          output_path,
                          in_parallel=4):

    if in_parallel == 1:
        for url, id in page_urls_list:
            scrape_flashcards_from_page(url, id,  output_path)

    else:
        #
        # creating the process pool
        pool = Pool(processes=in_parallel)

        results = pool.map(scrape_flashcards_helper, page_urls_list)

        return results


# In[ ]:

if NOTEBOOK:
    urls = [('http://www.studystack.com/Science&sortOrder=stars&page=1', 1),
            ('http://www.studystack.com/Science&sortOrder=stars&page=2', 2),
            ('http://www.studystack.com/Science&sortOrder=stars&page=3', 3),
            ('http://www.studystack.com/Science&sortOrder=stars&page=4', 4)]
    output_path = '../data/studystack.com/flashcards/'
    scrape_multiple_pages(urls,
                          output_path,
                          in_parallel=2)


# In[ ]:

def generate_url_pages_list(base_url, min_page_num, max_page_num):
    return [(base_url + str(i), i) for i in range(min_page_num, max_page_num)]


# In[ ]:

import argparse

BASE_URL = 'http://www.studystack.com/Physics&sortOrder=stars&page='
OUT_PATH = '../data/studystack.com/flashcards/Physics/'
N_PROCESSES = 1
MIN_PAG = 1
MAX_PAG = 10

if __name__ == '__main__':
    #
    # creating the parser for input args
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--n-processes', type=int,
                        default=N_PROCESSES,
                        help='number of parallel processes')

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
                        default=BASE_URL,
                        help='base url')

    # parsing the args
    args = parser.parse_args()

    #
    # generating page urls
    pages_urls = generate_url_pages_list(args.base_url, args.first_page, args.last_page + 1)

    #
    # starting scraping
    scrape_multiple_pages(pages_urls, args.output, in_parallel=args.n_processes)
