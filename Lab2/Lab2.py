import sqlite3
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
conn = sqlite3.connect("Chinook_Sqlite.sqlite")


def sql_overview():
    c = conn.cursor()
    for row in c.execute('SELECT * FROM Album'):
        print(row)
    for record in c.execute(
            'SELECT AlbumID, ArtistID, Title FROM Album WHERE AlbumID BETWEEN 10 AND 20 ORDER BY AlbumID DESC '):
        print(record)
    conn.close()


def task1():
    c = conn.cursor()
    for item in c.execute(
            "SELECT InvoiceId, CustomerId, BillingCity, Total FROM Invoice WHERE BillingCountry = 'USA' ORDER BY BillingCity DESC"):
        print(item)
    conn.close()

def task2():
    c = conn.cursor()
    for item in c.execute("SELECT Title, Name FROM Album INNER JOIN Artist ON Artist.ArtistID = Album.ArtistID"):
        print(item)
    # for item in c.execute("SELECT Title, Name FROM Album LEFT JOIN Artist ON Artist.ArtistID = Album.ArtistID"):
    #     print(item)
    # for item in c.execute("SELECT Title, Name FROM Artist LEFT JOIN Album ON Artist.ArtistID = Album.ArtistID"):
    #     print(item)

def rest_overview():
    req = requests.get("https://blockchain.info/ticker")
    bitcoin_dict = json.loads(req.text)
    print(bitcoin_dict)

def task3():
    req = requests.get("https://blockchain.info/ticker")
    bitcoin_dict = json.loads(req.text)
    bitcoin_df = pd.DataFrame.from_dict(bitcoin_dict, orient='index')
    print(bitcoin_df)

def open_weather_api():
    url = "https://api.openweathermap.org/data/2.5/onecall"
    api_key = "151f47236d5794b3db9309ad058c7749"
    latitude = 37.2431
    longitude = -115.7930
    req = requests.get(f"{url}?lat={latitude}&lon={longitude}&exclude=minutely&appid={api_key}")
    req_dict = json.loads(req.text)
    with open("data_file.json", "w") as write_file:
        json.dump(req_dict, write_file)
    print(req.text)

def task4():
    url = "https://api.openweathermap.org/data/2.5/onecall"
    api_key = "151f47236d5794b3db9309ad058c7749"
    latitude = 52.520008
    longitude = 13.404954
    req = requests.get(f"{url}?lat={latitude}&lon={longitude}&units={'metric'}&exclude=minutely&appid={api_key}")
    # print(req.text)
    req_dict = json.loads(req.text)
    with open("data_file_berlin.json", "w") as write_file:
        json.dump(req_dict, write_file)
    hourly_dict = req_dict['hourly']
    hourly_df = pd.DataFrame.from_dict(hourly_dict)
    hourly_df = hourly_df[['dt', 'temp', 'feels_like', 'humidity', 'wind_speed']]
    hourly_df['dt'] = pd.to_datetime(hourly_df['dt'], origin='unix', unit='s')
    print(hourly_df)
    hourly_df.plot(x='dt', y=['temp', 'feels_like', 'humidity'], subplots=3)
    plt.show()





if __name__ == '__main__':
    task4()
