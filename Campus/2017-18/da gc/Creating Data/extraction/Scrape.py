from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys

from difflib import get_close_matches

import urllib
from bs4 import BeautifulSoup

scheme=[]
agefund=[]
html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage=25&pgno=2&invObj=all&fundAge=%3C6&")
soup = BeautifulSoup(html,'html.parser')
num_page=str(soup.findAll("div", {"class" : "FL paging"})).count("</a>")
for j in range(0,num_page):
	html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage={}&pgno={}&invObj=all&fundAge=%3C6&".format(25*j,j+1))
	soup = BeautifulSoup(html,'html.parser')
	table=soup.findAll("tr",{"height":"20px"})
	for k in table:
		scheme.append(str(k.contents[1].contents[0].contents[0]))
		agefund.append(str(k.contents[-1].contents[0]))
with open("lessthan6mnths.csv","w+") as file:
	file.write("Names,Age of Fund (mth)\n")
	for j in range(len(scheme)):
		file.write(scheme[j]+","+agefund[j]+"\n")

scheme=[]
agefund=[]
html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage=25&pgno=2&invObj=all&fundAge=%3E6&")
soup = BeautifulSoup(html,'html.parser')
num_page=str(soup.findAll("div", {"class" : "FL paging"})).count("</a>")
for j in range(0,num_page):
	html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage={}&pgno={}&invObj=all&fundAge=%3E6&".format(25*j,j+1))
	soup = BeautifulSoup(html,'html.parser')
	table=soup.findAll("tr",{"height":"20px"})
	for k in table:
		scheme.append(str(k.contents[1].contents[0].contents[0]))
		agefund.append(str(k.contents[-1].contents[0]))
with open("GreaterThan6mnths.csv","w+") as file:
	file.write("Names,Age of Fund (mth)\n")
	for j in range(len(scheme)):
		print(scheme[j]+","+agefund[j]+"\n")
		try:
			file.write(scheme[j]+","+agefund[j]+"\n")
		except:
			continue

scheme=[]
agefund=[]
html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage=25&pgno=2&invObj=all&fundAge=%3E12&")
soup = BeautifulSoup(html,'html.parser')
num_page=str(soup.findAll("div", {"class" : "FL paging"})).count("</a>")
for j in range(0,num_page):
	html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage={}&pgno={}&invObj=all&fundAge=%3E12&".format(25*j,j+1))
	soup = BeautifulSoup(html,'html.parser')
	table=soup.findAll("tr",{"height":"20px"})
	for k in table:
		scheme.append(str(k.contents[1].contents[0].contents[0]))
		agefund.append(str(k.contents[-1].contents[0]))
with open("GreaterThan1year.csv","w+") as file:
	file.write("Names,Age of Fund (mth)\n")
	for j in range(len(scheme)):
		try:
			file.write(scheme[j]+","+agefund[j]+"\n")
		except:
			continue




scheme=[]
agefund=[]
html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage=25&pgno=2&invObj=all&fundAge=%3E24&")
soup = BeautifulSoup(html,'html.parser')
num_page=str(soup.findAll("div", {"class" : "FL paging"})).count("</a>")
for j in range(0,num_page):
	html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage={}&pgno={}&invObj=all&fundAge=%3E24&".format(25*j,j+1))
	soup = BeautifulSoup(html,'html.parser')
	table=soup.findAll("tr",{"height":"20px"})
	for k in table:
		scheme.append(str(k.contents[1].contents[0].contents[0]))
		agefund.append(str(k.contents[-1].contents[0]))
with open("GreaterThan2year.csv","w+") as file:
	file.write("Names,Age of Fund (mth)\n")
	for j in range(len(scheme)):
		try:
			file.write(scheme[j]+","+agefund[j]+"\n")
		except:
			continue

scheme=[]
agefund=[]
html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage=25&pgno=2&invObj=all&fundAge=%3E36&")
soup = BeautifulSoup(html,'html.parser')
num_page=str(soup.findAll("div", {"class" : "FL paging"})).count("</a>")
for j in range(0,num_page):
	html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage={}&pgno={}&invObj=all&fundAge=%3E36&".format(25*j,j+1))
	soup = BeautifulSoup(html,'html.parser')
	table=soup.findAll("tr",{"height":"20px"})
	for k in table:
		scheme.append(str(k.contents[1].contents[0].contents[0]))
		agefund.append(str(k.contents[-1].contents[0]))
with open("GreaterThan3year.csv","w+") as file:
	file.write("Names,Age of Fund (mth)\n")
	for j in range(len(scheme)):
		try:
			file.write(scheme[j]+","+agefund[j]+"\n")
		except:
			continue

scheme=[]
agefund=[]
html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage=25&pgno=2&invObj=all&fundAge=%3E60&")
soup = BeautifulSoup(html,'html.parser')
num_page=str(soup.findAll("div", {"class" : "FL paging"})).count("</a>")
for j in range(0,num_page):
	html=urllib.request.urlopen("http://nseindia.moneycontrol.com/mutualfundindia/advanced_search/commonsearch.php?head=&currentpage={}&pgno={}&invObj=all&fundAge=%3E60&".format(25*j,j+1))
	soup = BeautifulSoup(html,'html.parser')
	table=soup.findAll("tr",{"height":"20px"})
	for k in table:
		scheme.append(str(k.contents[1].contents[0].contents[0]))
		agefund.append(str(k.contents[-1].contents[0]))
with open("GreaterThan5year.csv","w+") as file:
	file.write("Names,Age of Fund (mth)\n")
	for j in range(len(scheme)):
		try:
			file.write(scheme[j]+","+agefund[j]+"\n")
		except:
			continue
