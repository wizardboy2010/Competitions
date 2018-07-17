import urllib
import re
from bs4 import BeautifulSoup

html=urllib.request.urlopen("http://www.moneycontrol.com/mf/mfinfo/amc_sch_listing.php?ffid=HD")
soup = BeautifulSoup(html,'html.parser')
print(soup.prettify)