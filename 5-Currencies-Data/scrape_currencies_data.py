import time
import pandas as pd
from selenium import webdriver

class SinaBookSpider(object):
    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.implicitly_wait(10)
    count = 0
    market_caps = []
    volumes = []
    tx_date = []
    open_price = []
    def run(self, currency_name):
        #time.sleep(5)
        #self.driver.get("https://www.google.com/")
        #time.sleep(5)
        self.driver.get("https://coinmarketcap.com/currencies/" + currency_name + "/historical-data/")
        for i in range(17):
            self.nextPage()
        self.parselist()
        while self.count == 0:
            self.parselist()
        print("market_caps:",self.market_caps)
        print("volumes:",self.volumes)
        print("tx_date:",self.tx_date)
        print("open_price:",self.open_price)
        total_data = pd.DataFrame([self.tx_date, self.open_price, self.volumes, self.market_caps])
        total_data = total_data.T
        total_data.columns=['tx_date', 'open_price', 'volume', 'market_caps']
        total_data.to_csv(currency_name + "_hist_data.csv")
        self.driver.quit()

    def parselist(self):
        time.sleep(5)
       
        #tabs = self.driver.find_element_by_tag_name('tbody')
        tds = self.driver.find_elements_by_tag_name('td')
        print(len(tds))
        self.count = 0
        for td in tds:
            if self.count%7 == 0:
                self.tx_date.append(td.text)
            if self.count%7 == 1:
                data = float(td.text.split("$")[1].replace(",",""))
                self.open_price.append(data)
            if self.count%7 == 6:
                data = float(td.text.split("$")[1].replace(",",""))
                self.market_caps.append(data)
            if self.count%7 == 5:
                data = float(td.text.split("$")[1].replace(",",""))
                self.volumes.append(data)
            self.count = self.count + 1
        print("self.count:",self.count)
    
    def nextPage(self):
        time.sleep(5)
        self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        self.driver.find_element_by_xpath('//button[normalize-space()="Load More"]').click()

if __name__ == '__main__':
    #"bitcoin", "ethereum","tether", "binance-coin",
    currencies_name = [  "cardano", "xrp", "dogecoin", "usd-coin", "polkadot-new","binance-usd"]
    spider = SinaBookSpider()

    for currency_name in currencies_name:
        spider.run(currency_name)

