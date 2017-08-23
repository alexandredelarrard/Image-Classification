# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:12:48 2017

@author: alexandre
"""

from selenium import webdriver 
from crawling_utils.crawling_functions import get_sitemap

class Crawling(object):
    
    def __init__(self, path_to_save_data, url, prod= False):
        self.path_to_save_data  = path_to_save_data
        self.website_url        = url
        
        if prod:
            self.driver = webdriver.PhantomJS()
        else:
            self.driver = webdriver.Firefox()
            
        self.parameters_website = {}
        
        self.liste_urls = get_sitemap(self.driver, path_to_save_data, url)

        self.driver.close()
    
        
if __name__ == "__main__":
    
    url = "https://www.loreal-paris.fr/"
    path_to_save = r"C:\Users\alexandre\Documents\qopius\git qopius\Image-Classification\saved_resources\save_crawling"
    
    cr = Crawling(path_to_save, url)
    cr.get_site_map(url)
    