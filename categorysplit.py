import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
import sys


df = pd.read_csv(sys.argv[1]) 
print(df.columns)
# url to be used for package
APP_LINK = "https://play.google.com/store/apps/details?id="
output_list = []; input_list = list(df['package'])

for pckg_name in input_list:
    # generate url, get html
    url = APP_LINK + pckg_name
    r = requests.get(url)

    if not (r.status_code==404):
        data = r.text
        soup = BeautifulSoup(data, 'html.parser')

        # parse result
        y = ""

        try:
            y = soup.find(itemprop = "genre").string
            print(y)
            #y = y.text
        except:
            y = "Unknown"
            print( "'Unknown' is stored for :"+pckg_name)

        output_list.append(y)
    else:
        y = "Unknown"
        print("'Unknown' is stored for :"+pckg_name)
        output_list.append(y)

df['app_category'] = output_list

df2 = df[['app','app_category','package','count']]

df2.to_csv(sys.argv[2])

print('Success! written to csv!')
