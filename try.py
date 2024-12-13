import requests  
import json  
import pandas as pd  
import time  
import os  
import re  

class CSGOMarketScraper:  
    def __init__(self):  
        self.prices_dir = 'item/prices'  
        os.makedirs(self.prices_dir, exist_ok=True)  

    def normalize_filename(self, name):  
        name = re.sub(r'[\\/*?:"<>|]', '', name)  
        name = name.replace(' ', '_')  
        name = name.replace('|', '-')  
        return name  

    def get_price_history(self, market_hash_name, base_delay=3):  
        url = f"https://steamcommunity.com/market/listings/730/{requests.utils.quote(market_hash_name)}"  
        headers = {  
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'  
        }  

        delay = base_delay  
        max_delay = 120  
        attempt = 1  

        while True:  
            try:  
                print(f"尝试第{attempt}次采集 {market_hash_name} 的价格历史，延迟 {delay} 秒...")  
                response = requests.get(url, headers=headers)  
                print(f"状态码: {response.status_code}")  

                if response.status_code == 200:  
                    line_pattern = r'var line1=(\[.*?\]);'  
                    match = re.search(line_pattern, response.text)  
                    if match:  
                        return json.loads(match.group(1))  

                if response.status_code == 429:  
                    print(f"请求受限，等待 {delay} 秒后重试...")  
                    time.sleep(delay)  
                    delay = min(delay * 2, max_delay)  
                    attempt += 1  
                    continue  

            except Exception as e:  
                print(f"请求失败或发生错误: {e}")  

            print(f"请求失败，等待 {delay} 秒后重试...")  
            time.sleep(delay)  
            delay = min(delay * 2, max_delay)  
            attempt += 1  

    def collect_price_histories(self, items_to_collect):  
        print("开始采集指定物品的价格历史数据...")  

        for idx, item_name in enumerate(items_to_collect):  
            print(f"\n处理物品 [{idx+1}/{len(items_to_collect)}]: {item_name}")  

            normalized_name = self.normalize_filename(item_name)  
            filename = f'{self.prices_dir}/{normalized_name}.csv'  

            if os.path.exists(filename):  
                print(f"✅ 已存在价格数据文件，跳过: {filename}")  
                continue  

            price_history = self.get_price_history(item_name)  
            if price_history:  
                # 调试输出：检查返回的原始数据  
                print(f"原始数据示例（前5条）：{price_history[:5]}")  

                # 转换为 DataFrame 并解析日期  
                df = pd.DataFrame(price_history, columns=['date', 'price', 'volume'])  

                # 调试输出：检查日期字段的原始格式  
                print(f"日期字段示例（前5条）：{df['date'].head()}")  

                # 解析日期  
                df['date'] = pd.to_datetime(df['date'].str.split('+').str[0].str.strip(), errors='coerce')  

                # 检查解析后的日期字段  
                print(f"解析后的日期字段示例（前5条）：{df['date'].head()}")  

                # 保存数据  
                df.to_csv(filename, index=False)  
                print(f"✅ 成功保存价格历史数据: {filename}")  
            else:  
                print(f"❌ 未能采集到价格历史数据: {item_name}")  

            time.sleep(3)  

if __name__ == "__main__":  
    items_to_collect = [  
        "AK-47 | Redline (Battle-Scarred)",  
        "AK-47 | Blue Laminate (Field-Tested)",  
        "AK-47 | Asiimov (Factory New)",  
        "AK-47 | Nightwish (Factory New)"  
    ]  

    scraper = CSGOMarketScraper()  
    scraper.collect_price_histories(items_to_collect)  
    print("\n所有指定物品的价格采集完成！")