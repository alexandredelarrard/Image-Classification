# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:14:18 2017

@author: alexandre
"""

from bs4 import BeautifulSoup
import unicodedata
from selenium import webdriver 
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import urllib2
import requests
from random import shuffle
import shutil
import glob
import pandas as pd
import numpy as np
import os

path_thor            = r'C:\Users\alexandre\Desktop\Tor Browser\Browser\TorBrowser\Data\Tor\torrc'
path_firefox         = r'C:\Users\alexandre\Desktop\Tor Browser\Browser\firefox.exe'
path_firefox_profile = r'C:\Users\alexandre\Desktop\Tor Browser\Browser\TorBrowser\Data\Browser\profile.default'


def get_sitemap(driver, path_to_save, url):

    informations = {}
    urls        = [url]
    alreadyseen = []
    
    while len(urls)>0:
        url_to_parse = urls[0]
        print(" parsing  : %s "%str(url_to_parse))
        
        new_liste, new_informations = get_urls(driver, url_to_parse, path_to_save)
        
        ### update dictionnary with new informations
        for k, v in new_informations.iteritems():
            informations[k] = informations.get(k, ()) + v
        
        alreadyseen += [url_to_parse]
        new_liste = [x for x in new_liste if x not in urls and x not in alreadyseen]
        urls = new_liste + urls[1:]
        np.sort(urls)
        
        print("Nbr remaining urls : %i, Nbr already seen : %i"%(len(urls), len(alreadyseen)))

    return informations

def get_urls(driver, url, path):
    
    current_level = {}
    
    #### get url
    driver = get_url(driver, url)
    
    #### create directories of url
    create_directory(url, path)
    
    ### if end of an url, parse all interesting informations
    if "?" in url:
        path_split = url.split("?")
        filters_liste = path_split[1].split("&")
        liste_paths = path_split[0].replace("https://", "").split("/")
        
        parsed_informations = parse_for_information(driver, url, path + "/" + path_split[0].replace("https://", ""))
        
        for filter_id in filters_liste:
            current_level[filter_id.split("=")[0]] = filter_id.split("=")[1]
        
        current_level["parsed_informations"] = parsed_informations

        for part in reversed(liste_paths):
            if part not in current_level:
                current_level = {part: current_level}
        
        print(current_level)
            
    ### parsing all unknown new urls
    parent = driver.find_elements_by_xpath("//a[@href]")
    liste_urls = []
    forbidenliste = ["/login", "/signin" , "#" , "/twitter", "/facebook", "/paiement", "bvstate=pg:2/ct:r"]
    
    for elem in parent:
        try:
            newurl = elem.get_attribute('href')
            if url in newurl and len([x for x in forbidenliste if x in newurl]) == 0:
                liste_urls.append(newurl)
                
        except Exception as e:
            print(e)
            pass
        
    if len(liste_urls)>0:
        liste_urls = pd.DataFrame(liste_urls)[0].value_counts().index
    else:
        print(url)
        
    return liste_urls, current_level
    
    
def parse_for_information(driver, url, path):
    
    output = {}
    
    #### get information of text from divs with class name
    liste_divs_to_parse = ['product-description', 'products-tabs-holder', 'flex-image--position']
    for divs in liste_divs_to_parse:
        output[divs]  = get_from_div_id(driver, "div[@class='%s']"%divs, path)
    
    return output


def get_url(driver, url):
    
    delay= 5
    try:
        driver.get(url)
        WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'main-header')))
        
    except TimeoutException:
        print("Loading took too much time!")
        
    except Exception as e:
        print(e)
        driver = change_ip(driver)
        print("driver has been changed and reseted")
        get_url(driver, url)
        
    return driver


def get_from_div_id(driver, div_id, path):
    
    parent = driver.find_element_by_xpath("//%s"%div_id)
    information = []
    
    if len(parent.text)>0:
        information = unicodedata.normalize('NFD', parent.text).encode('ascii', 'ignore')
        
    image = parent.find_elements_by_tag_name("img")
    if len(image)>0:
        for im in image:
            url_image = im.get_attribute("src")
            if "http" not in url_image:
                url_image = "https://" + url_image
                
            information += get_image(url_image, path)
    
    return information


def create_directory(url, path):
    
    ### uniquely create directory for a specific 
    elements_list = url.replace("https://", "").replace("http://", "").split("?")[0].split("/")
    path_add = ""
    
    try:
        for element_path in elements_list:
            if not os.path.exists('/'.join([path, path_add])):
                os.makedirs('/'.join([path, path_add]))
                
            path_add ='/'.join([path_add, element_path])
    
        if not os.path.exists('/'.join([path, path_add, "pictures"])):
            os.makedirs('/'.join([path, path_add, "pictures"]))
                
    except Exception as e:
        print("could not create folders for new url !!")
        print(e)
        pass
    
    
def get_image(url_image, path_to_save, filename =False):
    
    if filename:
        file_path = "/".join([path_to_save, '%s.jpg'%filename])
    else:    
        liste_files = glob.glob(path_to_save + "/*")
        nbr_files = len(liste_files)
        file_path = "/".join([path_to_save, '%i.jpg'%nbr_files])
        
    hdr = {'User-Agent': 'Mozilla/5.0 (Mobile; Windows Phone 8.1; Android 4.0; ARM; Trident/7.0; Touch; rv:11.0; IEMobile/11.0; Microsoft; Lumia 640 Dual SIM) like iPhone OS 7_0_3 Mac OS X AppleWebKit/537 (KHTML, like Gecko) Mobile Safari/537',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8',
           'Connection': 'keep-alive'}
        
    #### try with urllib2 
    try:
        req = urllib2.Request(url_image, headers=hdr)
        conn = urllib2.urlopen(req)
        output = open(file_path, 'wb')
        output.write(conn.read())
        output.close() 
        print("image saved !! ")
        
    #### if not working, try with req
    except urllib2.HTTPError:               
        req = requests.get(url_image, stream=True)
        if req.status_code == 200:
            with open(file_path, 'wb') as f:
                req.raw.decode_content = True
                shutil.copyfileobj(req.raw, f)
        else:
            print("could not download picture %s"%url_image)
        pass
    
    return '%i.jpg'%nbr_files

def change_ip(driver):
    
    """
    This function modifies meta data of browser for crawling. It enables to have a different IP adress
    - It is necessary to have firefox and thor downloaded on computer
    - It is based on country pool as well as a user_profile pool
    """
    
    ### if driver already open
    if driver !="": 
        driver.close()
    
    country =['AR','AU','PL', 'US', 'DE', 'IN', 'JP', 'CA' , 'LU' , 'BR', 'FI', 'HK', 'IT', 'AT','DK', 'IS', 'NZ' , 'NO' , 'PT' , 'SG']
    
    user_profile = ["Mozilla/5.0 (X11;     Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0   Safari/537.36", "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.112 Safari/537.36", "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)", "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) AppleWebKit/537.36 Gecko/20100101 Firefox/40.1", "Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) AppleWebKit/537.36 Gecko/20110201",
                    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36",
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
                    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9.0.1) Gecko/20080722 Firefox/3.0.1 Kapiko/3.0",
                    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
                    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_5_7; de-de) AppleWebKit/525.28.3 (KHTML, like Gecko) NetNewsWire/3.1.7",
                    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_5_6; de-de) AppleWebKit/525.27.1 (KHTML, like Gecko) NetNewsWire/3.1.7",
                    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_5_5; en-us) AppleWebKit/525.18 (KHTML, like Gecko) NetNewsWire/3.1.7"]
    
    shuffle(user_profile)
    shuffle(country)
    
    try:            
        f= open(path_thor, 'r')
        filedata = f.read()    
        
        if filedata[filedata.find("{") + 1:  filedata.find("}")] != country[0]:
            print("New country is %s"%country[0])
            newdata = filedata.replace(filedata[filedata.find("{") + 1:  filedata.find("}")], country[0])
        else:
            print("New country is %s"%country[1])
            newdata = filedata.replace(filedata[filedata.find("{") + 1:  filedata.find("}")], country[1])
        f.close()
                    
        f = open(path_thor,'w')
        f.write(newdata)
        f.close()
                
        binary = FirefoxBinary(path_firefox)    
        profile = FirefoxProfile(path_firefox_profile)
        profile.set_preference("general.useragent.override", user_profile[0])
        driver = webdriver.Firefox(firefox_profile= profile, firefox_binary= binary)
              
    except Exception:
        print(" has to change driver IP because of captcha ")
        driver = change_ip(driver)
        
    return driver   
            
#def get_element(driver, url, xpath, tag_name):
#    parent = driver.find_element_by_xpath("//div[@id='shelf-thumbs']")
#    reponse =[]
#    
#    for elmt in parent.find_elements_by_tag_name('article'):
#        soup = BeautifulSoup(elmt.get_attribute('innerHTML'), 'html.parser')
#        upc = elmt.get_attribute('data-rollup-id')
#        try:
#            price = float(soup.findAll('span',{"data-analytics-value":True})[0]["data-analytics-value"])
#        except Exception:
#            price = ''
#        try:
#            image = 'http:' + soup.findAll('img',{"data-original":True})[0]["data-original"]
#        except Exception:
#            image = 'http:' + soup.findAll('img',{"src":True})[0]["src"]
#        try:
#            print(soup.findAll('img',{"alt":True})[0]["alt"])
#            description =soup.findAll('img',{"alt":True})[0]["alt"].decode("latin-1").encode('ascii', 'ignore') 
#        except Exception:
#            description = ''
#            pass
#        
#        reponse.append([upc, price, description, image])
#        print([upc, price, description, image])
#        Ecriture_Image(site, image, upc, marque)
#    
#    return reponse

#    soup = BeautifulSoup(parent.get_attribute('innerHTML'), 'html.parser')
