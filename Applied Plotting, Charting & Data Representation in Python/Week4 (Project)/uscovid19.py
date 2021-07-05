import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
from bs4 import BeautifulSoup
import urllib.error
import ssl
from urllib.parse import urljoin
from urllib.parse import urlparse
from urllib.request import urlopen, Request

#get current US COVID19 Vaccination Data
url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/United%20States.csv'
s = requests.get(url).content

vac_df = pd.read_csv(io.StringIO(s.decode('utf-8')))

#CSV FROM DATA UPDATED UNTIL 06/18/2021
vaers2020_df = pd.read_csv('2020VAERSVAX.csv')
vaers2021_df = pd.read_csv('2021VAERSVAX.csv')

#get data as of 06/18/2021
vac_df = vac_df[vac_df['date'] == '2021-06-18']
vac_administered = vac_df['total_vaccinations'].values[0]

#get COVID19 adverse events from 2020
vaers2020_df = vaers2020_df[vaers2020_df['VAX_NAME'].str.contains('covid19', case = False)]

#get COVID19 adverse events from 2021
vaers2021_df = vaers2021_df[vaers2021_df['VAX_NAME'].str.contains('covid19', case = False)]

#get total COVID adverse event reports
vaers_count = vaers2020_df.shape[0] + vaers2021_df.shape[0]

#get adverse event proportion
prop_vae = vaers_count / vac_administered

print('Total Vaccines Administered:', '{:,}'.format(vac_administered))
print('Total VAERS Case Counts:', '{:,}'.format(vaers_count))
print('Vaccine Adverse Event Proportion:', "{:.4%}".format(prop_vae))


# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#url for current US population
url = 'https://countrymeters.info/en/United_States_of_America_(USA)'

#Attempt to access page info
#print('Retrieving current US population...')
try:
    document = urlopen(url, context = ctx)
    html = document.read()

    if document.getcode() != 200:
        print("Error on page: ", document.getcode())
    if 'text/html' != document.info().get_content_type() :
        print("Ignoring non-text/html page")

    soup = BeautifulSoup(html, "html.parser")
except:
    print("Unable to retrieve or parse page")

#get current US population
for td in soup.select('td'):
    if td.find('div', {'id' : 'cp1'}):
        pop = td.find('div', {'id' : 'cp1'}).text

print("Current US Population:", pop)


#url for current US COVID19 Cases and Deaths
url = 'https://www.worldometers.info/coronavirus/country/us/'

#Attempt to access page info
#print('Retrieving current US COVID19 Data...')

r = requests.get(url,
    headers = {"User-Agent" : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.97 Safari/537.36'})
soup = BeautifulSoup(r.text, 'html.parser')

#get current US COVID cases and deaths
temp = soup.find_all('div', {'class' : 'maincounter-number'})

cases = int(temp[0].text.strip().replace(",", ""))
deaths = int(temp[1].text.strip().replace(",", ""))
mort_rate = deaths / cases

print('Total COVID19 Cases in the U.S:', '{:,}'.format(cases))
print('Total COVID19 Deaths in the U.S:', '{:,}'.format(deaths))
print('U.S COVID19 Mortality Rate:', "{:.4%}".format(mort_rate))

xlabel = ['Total Vaccines Administered vs. Total VAERS Case Counts',
          'Total COVID19 Cases vs. Total COVID19 Deaths']
xval = [vaers_count, vac_administered, deaths, cases]


plt.figure()

plt.rcParams['axes.facecolor'] = 'silver'

plt.bar(xlabel[0], vaers_count, width = 0.3, color = 'orange')
plt.bar(xlabel[0], vac_administered, bottom = vaers_count, width = 0.3, color = 'b')
plt.bar(xlabel[1], deaths, width = 0.3, color = 'r')
plt.bar(xlabel[1], cases, bottom = deaths, width = 0.3, color = 'g')

plt.title('U.S COVID19 Data')
plt.ylabel('Count (in 100,000,000)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(['Total Vaccines Administered', 'Total VAERS Case Counts',
            'Total COVID19 Cases', 'Total COVID19 Deaths'], frameon = False)


plt.show()
