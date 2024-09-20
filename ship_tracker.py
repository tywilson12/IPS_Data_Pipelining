import requests
from lxml import etree
import time
import csv
from datetime import datetime
import pytz

def fetch_aurora_data():
    url = 'https://dot.alaska.gov/amhs/xml/Fleet.xml'
    response = requests.get(url)
    xml_data = response.content
    tree = etree.fromstring(xml_data)
    aurora = tree.xpath("//ship[@name='MV Aurora']")[0]
    latitude = aurora.find('.//lat').text
    longitude = aurora.find('.//lng').text
    speed = aurora.find('.//speed').text
    return latitude, longitude, speed

def write_data_to_csv(row_data):
    with open('MV_Aurora_Tracking.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

def main():
    headers = ['Time', 'Latitude', 'Longitude', 'Speed']
    with open('MV_Aurora_Tracking.csv', mode='a', newline='') as file:
        if file.tell() == 0:
            writer = csv.writer(file)
            writer.writerow(headers)

    alaska_tz = pytz.timezone('America/Anchorage')
    
    while True:
        latitude, longitude, speed = fetch_aurora_data()
        current_time = datetime.now(alaska_tz).strftime('%Y-%m-%d %H:%M:%S')
        write_data_to_csv([current_time, latitude, longitude, speed])

        time.sleep(60)

if __name__ == "__main__":
    main()