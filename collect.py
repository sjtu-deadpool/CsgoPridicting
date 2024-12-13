import requests
import json
import pandas as pd
import time
from datetime import datetime
import os
import re

class CSGOMarketScraper:
   def __init__(self):
       # 创建数据目录
       self.data_dir = 'csgo_market_data'
       self.items_dir = f'{self.data_dir}/items'
       self.prices_dir = f'{self.data_dir}/prices'
       os.makedirs(self.items_dir, exist_ok=True)
       os.makedirs(self.prices_dir, exist_ok=True)
       
   def normalize_filename(self, name):
       """规范化文件名"""
       # 移除非法字符,替换特殊字符
       name = re.sub(r'[\\/*?:"<>|]', '', name)
       name = name.replace(' ', '_')
       name = name.replace('|', '-')
       return name
   
   def get_market_items(self, start=0, count=100, base_delay=2):
       """获取市场物品列表，指数级增加延迟直到成功"""
       url = "https://steamcommunity.com/market/search/render/"
       params = {
           'appid': 730,
           'norender': 1,
           'start': start,
           'count': count,
           'currency': 1
       }
       headers = {
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
       }
       
       delay = base_delay
       max_delay = 120  # 最大延迟2分钟
       attempt = 1
       
       while True:
           try:
               print(f"尝试第{attempt}次, 延迟{delay}秒...")
               response = requests.get(url, params=params, headers=headers)
               print(f"状态码: {response.status_code}")
               
               if response.status_code == 200:
                   data = response.json()
                   return data.get('results', []), data.get('total_count', 0)
               
               # 429表示请求过多
               if response.status_code == 429:
                   print(f"请求受限，等待{delay}秒后重试...")
                   time.sleep(delay)
                   delay = min(delay * 2, max_delay)  # 指数增加但不超过最大值
                   attempt += 1
                   continue
                   
           except Exception as e:
               print(f"错误: {e}")
           
           print(f"请求失败，等待{delay}秒后重试...")
           time.sleep(delay)
           delay = min(delay * 2, max_delay)
           attempt += 1

   def get_price_history(self, market_hash_name, base_delay=3):
       """获取物品价格历史，带指数级重试"""
       url = f"https://steamcommunity.com/market/listings/730/{requests.utils.quote(market_hash_name)}"
       headers = {
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
       }
       
       delay = base_delay
       max_delay = 120
       attempt = 1
       
       while True:
           try:
               print(f"尝试第{attempt}次, 延迟{delay}秒...")
               response = requests.get(url, headers=headers)
               print(f"状态码: {response.status_code}")
               
               if response.status_code == 200:
                   line_pattern = r'var line1=(\[.*?\]);'
                   match = re.search(line_pattern, response.text)
                   if match:
                       return json.loads(match.group(1))
                       
               if response.status_code == 429:
                   print(f"请求受限，等待{delay}秒后重试...")
                   time.sleep(delay)
                   delay = min(delay * 2, max_delay)
                   attempt += 1
                   continue
                   
           except Exception as e:
               print(f"错误: {e}")
           
           print(f"请求失败，等待{delay}秒后重试...")
           time.sleep(delay)
           delay = min(delay * 2, max_delay)
           attempt += 1
   
   def collect_all_items(self):
       """收集所有物品信息"""
       print("开始收集物品信息...")
       
       start = 0
       items_list = []
       total_items = 1
       
       while start < total_items:
           items, total_count = self.get_market_items(start=start)
           total_items = total_count
           
           if items:
               items_list.extend(items)
               print(f"已获取 {len(items_list)}/{total_items} 个物品")
               
               # 保存进度
               df = pd.DataFrame(items_list)
               df.to_csv(f'{self.items_dir}/items_list.csv', index=False)
               
               start += 100
               # 基础延迟2秒
               time.sleep(2)
       
       return items_list
   
   def collect_price_histories(self, items_list):
       """收集所有物品的价格历史"""
       print("\n开始收集价格历史...")
       
       for idx, item in enumerate(items_list):
           name = item['name']
           normalized_name = self.normalize_filename(name)
           
           print(f"\n处理 [{idx+1}/{len(items_list)}] {name}")
           
           # 检查是否已经存在
           filename = f'{self.prices_dir}/{normalized_name}.csv'
           if os.path.exists(filename):
               print("已存在,跳过")
               continue
           
           price_history = self.get_price_history(name)
           if price_history:
               # 转换为DataFrame并保存
               df = pd.DataFrame(price_history, columns=['date', 'price', 'volume'])
               df['date'] = pd.to_datetime(df['date'].str.split('+').str[0].str.strip())
               df.to_csv(filename, index=False)
               print(f"保存成功: {len(price_history)} 条记录")
           else:
               print("获取价格历史失败")
           
           time.sleep(3)  # 基础延迟

def main():
   scraper = CSGOMarketScraper()
   
   # 获取所有物品
   print("第1步：收集所有物品信息")
   items_list = scraper.collect_all_items()
   print(f"\n共获取 {len(items_list)} 个物品")
   
   # 获取所有价格历史
   print("\n第2步：收集所有物品的价格历史")
   scraper.collect_price_histories(items_list)
   print("\n数据收集完成!")

if __name__ == "__main__":
   main()