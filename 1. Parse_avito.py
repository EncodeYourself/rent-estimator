#%%
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
import numpy as np
from os import getcwd
import re
import time

#%%
class avito_parser:
    def __init__(self, url, page = 1, download_driver = False):
        self.starting_page = page
        self.current_page = page
        self.base_url = url
        s = Service(r'/home/illusion/.wdm/drivers/geckodriver/linux64/0.31/geckodriver')
        self.database = getcwd() + '/database.csv'
        self.driver = webdriver.Firefox(service=s)
        #self.driver.maximize_window()
        self.last_page = None
        self.applicant = None


    def parse(self, url = None):
        if url is None:
            url= self.base_url + str(self.current_page)
        try:
            self.driver.get(url)
        except Exception:
            self.driver.delete_all_cookies()
            self.parse(url)

    def get_items(self):
        return self.driver.find_elements(
            By.CSS_SELECTOR, 'div[data-marker="item"')

    def get_geo_info(self, item):
        try:
            geo_info = item.find_element(
                By.CLASS_NAME, "geo-georeferences-SEtee").text

            try:
                item.find_element(By.CLASS_NAME, "geo-icons-uMILt")
                metro = re.sub(r'\d*|-|\.|мин|–',r'', geo_info)
                metro = re.sub(r'(\S)(от|до) ',r'\1', metro)

                distance = re.sub(r'[^\d–]',r'', geo_info)
                if distance == '':
                    distance = 40

                district = np.nan

            except:
                district = geo_info
                metro = np.nan
                distance = np.nan

        except:
            district = item.find_element(
                By.CLASS_NAME, "geo-address-fhHd0").text
            district = district.split(',')[0]
            metro = np.nan
            distance = np.nan

        return metro, distance, district

    def get_basic_info(self, item):
        original_title = item.find_element(
            By.CLASS_NAME, 'title-root-zZCwT').text

        try:
            title = re.sub(r'(\d),([\d]?)',r'\1.\2', original_title)
            title = re.sub(r'(\sм²| эт.)', r'', title)
            title = re.sub(r'/', r', ', title)
            app_type, size, floor, maxfloor = title.split(', ')
            whole_string = np.nan

        except:
            app_type = np.nan
            size = np.nan
            floor = np.nan
            maxfloor = np.nan
            whole_string = original_title

        return app_type, size, floor, maxfloor, whole_string

    def get_price(self, item):
        price = item.find_element(
            By.CSS_SELECTOR, 'span[data-marker="item-price"]').text
        price = re.sub(r'\D', '', price)

        return price

    def next_page(self):
        self.driver.find_element(
            By.CSS_SELECTOR, 'span[data-marker="pagination-button/next"]').click()


    def extract_and_save(self):
        items = self.get_items()
        for item in items:
            app_type, size, floor, maxfloor, whole_string = self.get_basic_info(item)
            price = self.get_price(item)
            metro, distance, district = self.get_geo_info(item)

            self.write([app_type, size, floor, maxfloor, metro, distance,
                        district, self.applicant, price, whole_string])

    def create_csv(self):
        with open(self.database, 'w') as file:
            csv.writer(file).writerow(
                ['Type', 'm2_size', 'floor', 'maxfloor', 'metro', 'district',
                 'distance', 'sender', 'price'])

    def write(self, array):
        with open(self.database, 'a') as file:
            csv.writer(file).writerow(array)
    def wait(self):
        time.sleep(np.random.randint(7, 12))
       
    def loop(self, page = None, create_new_csv = True,
             applicant_switch_pattern = 'both'):
        assert applicant_switch_pattern in ['both', 'owner', 'agency'], \
            'The applicant switch patter can only be "both", "owner" or "agency"' 
            
        if create_new_csv == True:
            self.create_csv()
        self.parse()
        if applicant_switch_pattern == 'both':
            time.sleep(3)
            self.pretend_to_be_human(2)
            self.switch_applicant(applicant = 'owner')
            self.run()
            self.switch_applicant(applicant = 'agency')
            self.run()
        else:
            time.sleep(3)
            self.pretend_to_be_human(2)
            self.switch_applicant(applicant = applicant_switch_pattern)
            self.run()

    def run(self):
        self.get_last_page()
        print(self.last_page, ' is the last page')
        self.extract_and_save()
        print(f'Starting page {self.starting_page} is extracted')
        for page in range(self.starting_page, self.last_page - self.starting_page + 1):
            print(f'Parsing #{page} page is completed')
            self.pretend_to_be_human()
            self.wait()
            self.extract_and_save()
            print('Extracted')
            self.next_page()
            
    def switch_applicant(self, applicant = 'owner'):
        assert applicant in ['owner', 'agency'], \
            'Applicant can either be "owner" or "agency"!'

        if applicant == 'owner':
            self.applicant = 'Собственник'
            xpath = '//*[contains(text(), "Частные")]'
        elif applicant == 'agency':
            self.applicant = 'Агентство'
            xpath = '//*[contains(text(), "Агентства")]'

        self.driver.find_element(By.XPATH, xpath).click()
        self.driver.find_element(
            By.CSS_SELECTOR, 'button.button-primary-x_x8w:nth-child(1)').click()
        self.page = 1

    def get_last_page(self):
        raw_pages = self.driver.find_element(
            By.CSS_SELECTOR, 'div[data-marker="pagination-button"]').text
        self.last_page = int(re.sub(r'[^0-9]+', '', raw_pages)[5:])
        
    def pretend_to_be_human(self, number = None, pattern = 0):
        '''Had a couple of ideas in order to avoid being banned, but it does
        seem the avito anti-bot systems are... lacking, so this part was not 
        implemented'''
        if number is None:
            number = np.random.randint(1,4)
        if pattern == 0:
            action = ActionChains(self.driver)
            for repeats in range(number):
                action.send_keys(Keys.PAGE_DOWN)
                action.perform()
        else:
            pass