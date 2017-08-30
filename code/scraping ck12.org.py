
# coding: utf-8

# In[19]:

#
# science categories
#
# ['earth science', 'life science', 'physical science', 'biology', 'chemistry', 'physics']
import pandas

# http://www.ck12.org/book/CK-12-Physics-Concepts-Intermediate/
# -> get the sections
BASE_URL = 'http://www.ck12.org/'
BOOK_PATH = 'book/'
book_title = 'CK-12-Physics-Concepts-Intermediate/'
section = ''
# # http://www.ck12.org/book/CK-12-Physics-Concepts-Intermediate/section/2.0/


#
# get the section
# http://www.ck12.org/book/CK-12-Physics-Concepts-Intermediate/section/1.1/
#
# get the 'read' portion
# body/#contentwrap/../#artifact_content


from selenium import webdriver
import time
from time import perf_counter

PHANTOMJS_PATH = '/usr/bin/phantomjs'


def get_section_text_content(section_url,
                             driver=None,
                             content_id='artifact_content',
                             remove_questions=False,
                             waiting_time=2,
                             phantomjs_path=PHANTOMJS_PATH,
                             close=False):
    #
    # initing phantomjs driver
    # TODO: I have to turn this into a context manager...
    if not driver:
        driver = webdriver.PhantomJS(executable_path=phantomjs_path)

    get_start_t = perf_counter()
    #
    # making the request
    driver.get(section_url)
    get_end_t = perf_counter()
    print('\t\t\tgot request in {} secs'.format(get_end_t - get_start_t), end='       \r')

    #
    # waiting some time to let it render
    time.sleep(waiting_time)

    extract_start_t = perf_counter()
    #
    # remove ol (containing questions?)
    if remove_questions:
        rem_ol_js = """var ols = document.getElementsByTagName('ol');
                    for (index = ols.length - 1; index >= 0; index--) {
                        ols[index].parentNode.removeChild(ols[index]);
                    }"""
        driver.execute_script(rem_ol_js)

    #
    # retrieve the content section text
    content_text = driver.find_element_by_id(content_id).text
    extract_end_t = perf_counter()
    print('\t\t\textracted content in {} secs'.format(
        extract_end_t - extract_start_t),  end='       \r')

    #
    # closing
    if close:
        driver.close()

    return content_text


def get_sections_urls(book_url,
                      driver=None,
                      toc_class='toc_list',
                      waiting_time=2,
                      phantomjs_path=PHANTOMJS_PATH,
                      close=False):
    #
    # initing phantomjs driver
    if not driver:
        driver = webdriver.PhantomJS(executable_path=phantomjs_path)
    #
    # making the request
    driver.get(book_url)
    #
    # waiting some time to let it render
    time.sleep(waiting_time)

    toc_content = driver.find_element_by_class_name(toc_class)
    # print(toc_content.hrefs)
    toc_urls = [anchor.get_attribute("href")
                for anchor in toc_content.find_elements_by_partial_link_text('.')]
    # print(toc_urls)

    #
    # trying at the whole level, some pages do not have a toc list section
    if not toc_urls:
        print('\t\t\t\t\ttrying to get link in all the page')
        toc_urls = [anchor.get_attribute("href")
                    for anchor in driver.find_elements_by_partial_link_text('.')]

    #
    # closing
    if close:
        driver.close()

    return toc_urls


def bookitem_list_to_csv(bookitems, output_path, separator=',', header=['state', 'subject', 'grade', 'url']):

    with open(output_path, 'w') as csv_file:
        #
        # write header
        if header:
            header_line = separator.join(header) + '\n'
            csv_file.write(header_line)

        for bi in bookitems:
            book_line = separator.join(
                [str(e) for e in [bi.state, bi.subject, bi.grade, bi.url]]) + '\n'
            csv_file.write(book_line)
            csv_file.flush()


from collections import namedtuple
from time import perf_counter

BookItem = namedtuple('BookItem', ['state', 'subject', 'grade', 'url'])

STATE_LIST = ['CCSS', 'NGSS', 'NSES', 'NCTM',
              'Alabama', 'Alaska', 'Arizona', 'Arkansas',
              'California', 'Colorado', 'Connecticut',
              'Delaware', 'District of Columbia',
              'Florida', 'Georgia', 'Hawaii',
              'Idaho', 'Illinois', 'Indiana', 'Iowa',
              'Kansas', 'Kentucky',
              'Louisiana',
              'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
              'Missouri', 'Montana',
              'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
              'North Carolina', 'North Dakota',
              'Ohio', 'Oklahoma', 'Oregon',
              'Pennsylvania', 'Rhode Island',
              'South Carolina', 'South Dakota',
              'Tennessee', 'Texas', 'Utah',
              'Vermont', 'Virginia',
              'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']


def generate_urls_by_select_combinations(page_url,
                                         driver=None,
                                         state_id='standards_state',
                                         subject_id='standards_subject',
                                         grade_id='standards_grade',
                                         booklist_id='standardBoardLinks',
                                         state_list=STATE_LIST,
                                         waiting_time=2,
                                         phantomjs_path=PHANTOMJS_PATH,
                                         close=False):

    book_urls = []

    #
    # initing phantomjs driver
    if not driver:
        driver = webdriver.PhantomJS(executable_path=phantomjs_path)
    #
    # making the request
    driver.get(page_url)
    #
    # waiting some time to let it render
    time.sleep(waiting_time)

    state_combo = driver.find_element_by_id(state_id)
    #
    # for each possible state:
    for state_option in state_combo.find_elements_by_tag_name('option'):

        state_starting_t = perf_counter()
        state_value = state_option.text
        if state_value in state_list:
            # if state_value == 'Arkansas':
            # print('state option:', state_value)
            #
            # simulate a click on the country combo-box
            state_option.click()
            #
            # get the available subjects
            subject_combo = driver.find_element_by_id(subject_id)

            for sub_option in subject_combo.find_elements_by_tag_name('option'):

                subject_starting_t = perf_counter()
                subject_value = sub_option.text
                if subject_value != 'Subject':
                    # print('\tsubject option:', subject_value)

                    #
                    # simulate a click on the subject now
                    sub_option.click()
                    #
                    # get the grade
                    grade_combo = driver.find_element_by_id(grade_id)
                    for gra_option in grade_combo.find_elements_by_tag_name('option'):
                        #
                        # simulate even this click
                        grade_value = gra_option.text
                        if grade_value != 'Grade':
                            # print('\t\tgrade option:', grade_value)

                            gra_option.click()

                            time.sleep(waiting_time)
                            #
                            # retrieve the book list part, if present
                            #

                            try:
                                booklist = driver.find_element_by_id(booklist_id)
                                anchors = booklist.find_elements_by_xpath(
                                    '//a[following-sibling::h2]')
                                for a in anchors:
                                    url_value = a.get_attribute('href')
                                    # print('\t\t\t', url_value)
                                    print('{0}\t{1}\t{2}\t{3}'.format(state_value,
                                                                      subject_value,
                                                                      grade_value,
                                                                      url_value),
                                          end='       \r')

                                    #
                                    # storing it
                                    book_urls.append(BookItem(state=state_value,
                                                              subject=subject_value,
                                                              grade=grade_value,
                                                              url=url_value))
                            except:
                                print('\rNo books for {0} {1} {2}'.format(
                                    state_value, subject_value, grade_value))

                    subject_end_t = perf_counter()
                    print('\r{2}\t{0} completed! in {1} secs'.format(subject_value,
                                                                     subject_end_t -
                                                                     subject_starting_t,
                                                                     state_value).ljust(20))

            state_end_t = perf_counter()
            print('\r{0} completed in {1} secs!'.format(
                state_value, state_end_t - state_starting_t).ljust(20))

    #
    # closing
    if close:
        driver.close()

    return book_urls


from collections import deque

state_list = ['Alabama', 'Alaska', 'Arizona',
              'Arkansas', 'California', 'Colorado',
              'Connecticut', 'Delaware', 'District of Columbia',
              'Florida', 'Georgia', 'Hawaii',
              'Idaho', 'Illinois', 'Indiana',
              'Iowa', 'Kansas', 'Kentucky',
              'Louisiana', 'Maine', 'Maryland',
                           'Massachusetts', 'Michigan', 'Minnesota',
                           'Mississippi', 'Missouri', 'Montana',
                           'Nebraska', 'Nevada', 'New Hampshire',
                           'New Jersey', 'New Mexico', 'New York',
                           'North Carolina', 'North Dakota', 'Ohio',
                           'Oklahoma', 'Oregon', 'Pennsylvania',
                           'Rhode Island', 'South Carolina', 'South Dakota',
                           'Tennessee', 'Texas', 'Utah',
                           'Vermont', 'Virginia', 'Washington',
                           'West Virginia', 'Wisconsin', 'Wyoming']


def scrape_state_book_list(starting_url,
                           state_list,
                           waiting_time=7,
                           n_at_a_time=1,
                           output_path='../tmp/booklist.csv'):

    states_to_process = deque(state_list)

    book_urls = []
    while states_to_process:
        #
        # get a state to process
        states = [states_to_process.popleft() for _i in range(n_at_a_time)]
        #
        # call the method
        try:
            book_urls.extend(generate_urls_by_select_combinations(starting_url,
                                                                  waiting_time=waiting_time,
                                                                  state_list=states))
        except:
            print('An error occurred!\nTrying again for states', states)

            for state in states:
                states_to_process.appendleft(state)

    #
    # at the end, make the csv file
    bookitem_list_to_csv(book_urls, output_path)


import os
from selenium import webdriver
import time

PHANTOMJS_PATH = '/usr/bin/phantomjs'


def scrape_book(book_url,
                output_path,
                remove_questions=False,
                waiting_time=5):

    # driver = webdriver.PhantomJS(executable_path=PHANTOMJS_PATH)

    #
    # from book url, get all sections urls
    section_urls = get_sections_urls(book_url, waiting_time=waiting_time)

    #
    #
    print('got {0} sections: [{1} -> {2}]'.format(len(section_urls),
                                                  section_urls[0],
                                                  section_urls[-1]))
    for section in section_urls:

        section_id = section.split("/")[-2]
        section_path = output_path + section_id + '/'
        print('Processing section', section_id)

        driver = webdriver.PhantomJS(executable_path=PHANTOMJS_PATH)

        #
        # getting subsections
        subsection_urls = get_sections_urls(section, driver=driver, waiting_time=waiting_time)
        print('\tgot {0} subsections: [{1} -> {2}]'.format(len(subsection_urls),
                                                           subsection_urls[0],
                                                           subsection_urls[-1]))

        index_path = section_path + 'url_index.txt'
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        with open(index_path, 'w') as subsection_index:
            for subsection in subsection_urls:
                subsection_index.write(subsection + '\n')

        driver.close()


# scrape_book('http://www.ck12.org/book/CK-12-Middle-School-Math-Grade-6/',
#             '../data/ck12.org/books/CK-12-Middle-School-Math-Grade-6/')


# # In[8]:

from collections import deque


def scrape_books(book_urls,
                 output_path,
                 waiting_time=5):

    #
    # a queue to process items
    books_to_process = deque(book_urls)

    while books_to_process:

        url = books_to_process.popleft()

        try:
            #
            # create path
            book_name = url.split('/')[-2]
            print('**** processing book {} ****'.format(book_name))
            book_out_path = output_path + book_name + '/'
            scrape_book(url, book_out_path, waiting_time=waiting_time)
        except:
            print('++++ some error occurred, retrying for book {} +++'.format(url))
            books_to_process.appendleft(url)


# In[9]:

# already_done_books = ['http://www.ck12.org/book/CK-12-Algebra-I-Concepts-Honors/',
#                       'http://www.ck12.org/book/CK-12-Algebra-II-with-Trigonometry-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Algebra-I-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Basic-Algebra-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Basic-Geometry-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Middle-School-Math-Concepts-Grade-6/',
#                       'http://www.ck12.org/book/CK-12-Middle-School-Math-Concepts-Grade-7/',
#                       'http://www.ck12.org/book/CK-12-Middle-School-Math-Concepts-Grade-8/',
#                       'http://www.ck12.org/book/CK-12-Middle-School-Math-Grade-8/',
#                       'http://www.ck12.org/book/CK-12-Precalculus-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Physical-Science-Concepts-For-Middle-School/',
#                       'http://www.ck12.org/book/CK-12-Math-Analysis-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Algebra-Basic/',
#                       'http://www.ck12.org/book/CK-12-Middle-School-Math-Grade-6/',
#                       'http://www.ck12.org/book/CK-12-Middle-School-Math-Grade-7/',
#                       'http://www.ck12.org/book/CK-12-Calculus-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Geometry-Concepts-Honors/',
#                       'http://www.ck12.org/book/CK-12-Geometry-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Foundation-and-Leadership-Public-Schools-College-Access-Reader%253A-Geometry/',
#                       'http://www.ck12.org/book/CK-12-Geometry-Second-Edition/',
#                       'http://www.ck12.org/book/CK-12-Advanced-Probability-and-Statistics-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Basic-Probability-and-Statistics-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Probability-and-Statistics-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Trigonometry-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Physics-Concepts-Intermediate/',
#                       'http://www.ck12.org/book/Peoples-Physics-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Physics---Intermediate/',
#                       'http://www.ck12.org/book/CK-12-Biology-Advanced-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Biology-Concepts/',
#                       'http://www.ck12.org/book/CK-12-Earth-Science-Concepts-For-Middle-School/',
#                       'http://www.ck12.org/book/CK-12-Chemistry-Concepts-Intermediate/',
#                       'http://www.ck12.org/book/CK-12-Earth-Science-Concepts-For-High-School/',
#                       'http://www.ck12.org/book/CK-12-Life-Science-Concepts-For-Middle-School/',
#                       'http://www.ck12.org/book/CK-12-Biology/',
#                       'http://www.ck12.org/book/CK-12-Earth-Science-For-Middle-School/',
#                       'http://www.ck12.org/book/CK-12-Calculus/',
#                       'http://www.ck12.org/book/CK-12-Life-Science-For-Middle-School/',
#                       ]


# unaccessible_books = [
#     'http://www.ck12.org/book/CK-12-Algebra-I-Second-Edition/',
#     'http://www.ck12.org/book/CK-12-Geometry-Basic/',
#     'http://www.ck12.org/book/CK-12-Basic-Probability-and-Statistics-A-Short-Course/'
# ]

# books_still_to_do = [url for url in unique_urls if url not in set(
#     already_done_books).union(set(unaccessible_books))]
# print('there are still {} books to process', len(books_still_to_do))
# scrape_books(books_still_to_do,
#              '../data/ck12.org/books/', waiting_time=7)


# # In[3]:

#
# collecting now index files
import glob
from collections import deque
from selenium import webdriver
import os


def scrape_from_index_file(index_path,
                           section_path,
                           waiting_time=5,
                           remove_questions=False):

    with open(index_path, 'r') as index_file:

        section_urls = [url.rstrip() for url in index_file.readlines()]
        # print('\t--> There are {0} sections to process [from {1} to {2}]'.format(len(section_urls),
        #                                                                        section_urls[0],
        #                                                                         section_urls[-1]))

        #
        # now trying to scrape each section
        sections_to_process = deque(section_urls)

        driver = webdriver.PhantomJS(executable_path=PHANTOMJS_PATH)
        driver.set_page_load_timeout(30)

        while sections_to_process:

            url = sections_to_process.popleft()

            try:
                subsection_id = url.split('/')[-2]
                subsection_path = section_path + subsection_id + '.txt'

                if not os.path.exists(subsection_path):
                    subsection_text = get_section_text_content(url,
                                                               driver=driver,
                                                               waiting_time=waiting_time,
                                                               remove_questions=remove_questions)

                    # subsection_path = section_path + subsection_id + '.txt'
                    print('\t\tWriting to file', subsection_id, end='      \r')
                    os.makedirs(os.path.dirname(subsection_path), exist_ok=True)

                    with open(subsection_path, 'w') as subsection_file:
                        subsection_file.write(subsection_text)
                else:
                    print('path already exists, skipping!')

            # except Exception as e:
            except:
                # print(str(e))
                print('\t+++ Error processing {} retrying +++'.format(url))
                time.sleep(5)
                driver.close()
                driver = webdriver.PhantomJS(executable_path=PHANTOMJS_PATH)
                driver.set_page_load_timeout(30)

                sections_to_process.appendleft(url)

        driver.close()


# In[3]:

# index_path = '/home/valerio/Petto Redigi/kaggle/allen-ai-science-qa/data/ck12.org/books/CK-12-Geometry-Concepts-Honors/1.0/url_index.txt'
# section_path = '/'.join(index_path.split('/')[:-1]) + '/'
# print(section_path)
# scrape_from_index_file(index_path, section_path)


# # In[5]:

import glob


def scrape_contents_from_all_index_files(base_path,
                                         index_file='*/*/url_index.txt'):
    # print(base_path + index_file)
    index_paths = glob.glob(base_path + index_file)
    print('There are {} index files to process'.format(len(index_paths)))

    indexes_to_process = deque(sorted(index_paths))

    i = 0
    while indexes_to_process:
        try:
            index = indexes_to_process.popleft()
            print('\n**** [{0}/{1}] Processing {2} ****'.format(i + 1,
                                                                len(index_paths),
                                                                index))
            section_path = '/'.join(index.split('/')[:-1]) + '/'
            scrape_from_index_file(index, section_path, waiting_time=4)
            i += 1
        except:
            print('\t°°° Error processing {} retrying °°°'.format(index))
            time.sleep(5)
            indexes_to_process.appendleft(index)

    # for i, index in enumerate(sorted(index_paths)):
    #     print('\n**** [{0}/{1}] Processing {2} ****'.format(i + 1,
    #                                                         len(index_paths),
    #                                                         index))
    #     section_path = '/'.join(index.split('/')[:-1]) + '/'
    #     scrape_from_index_file(index, section_path, waiting_time=4)


# In[ ]:

def scrape_contents(output_path='../tmp'):

    os.makedirs(output_path, exist_ok=True)
    starting_url = 'http://www.ck12.org/standards/Life%20Science/US.DC/11'
    out_csv_path = os.path.join(output_path, 'booklist.csv')

    scrape_state_book_list(starting_url,
                           state_list,
                           waiting_time=7,
                           n_at_a_time=1,
                           output_path=out_csv_path)

    bookitems_frame = pandas.read_csv(out_csv_path)
    print(bookitems_frame.head(10))

    unique_urls = bookitems_frame['url'].unique()
    print('There are only {} urls'.format(len(unique_urls)))

    print(unique_urls)

    #
    # storing for resuming later, it hungs up often
    list_output = os.path.join(output_path, 'bookurls.txt')
    with open(list_output, 'w') as url_log:
        for url in unique_urls:
            url_log.write(url + '\n')

    book_out_path = os.path.join(output_path, 'books')
    os.makedirs(book_out_path, exist_ok=True)
    scrape_books([book for book in unique_urls],
                 book_out_path, waiting_time=7)

    # base_url = os.path.join(output_path, 'books')
    # os.makedirs(base_url, exist_ok=True)

    scrape_contents_from_all_index_files(book_out_path)


# In[ ]:

if __name__ == '__main__':
    scrape_contents()
