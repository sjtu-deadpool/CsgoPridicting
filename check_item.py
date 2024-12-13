import os  
import re  

class PriceChecker:  
    def __init__(self, prices_dir):  
        self.prices_dir = prices_dir  

    def normalize_filename(self, name):  
        """规范化文件名，与采集时的命名规则保持一致"""  
        name = re.sub(r'[\\/*?:"<>|]', '', name)  # 移除非法字符  
        name = name.replace(' ', '_')  # 替换空格为下划线  
        name = name.replace('|', '-')  # 替换竖线为连字符  
        return name  

    def check_prices(self, items_to_check):  
        """检查文件夹中是否包含指定物品的价格文件"""  
        print(f"检查目录: {self.prices_dir}")  
        if not os.path.exists(self.prices_dir):  
            print("价格目录不存在！")  
            return  

        # 获取目录中所有文件的文件名  
        existing_files = os.listdir(self.prices_dir)  
        print(f"找到 {len(existing_files)} 个文件")  

        # 检查每个物品是否存在对应的价格文件  
        for item in items_to_check:  
            normalized_name = self.normalize_filename(item)  
            filename = f"{normalized_name}.csv"  
            if filename in existing_files:  
                print(f"✅ 找到价格文件: {filename}")  
            else:  
                print(f"❌ 未找到价格文件: {filename}")  

if __name__ == "__main__":  
    # 定义价格文件夹路径  
    prices_dir = "item/prices"  

    # 定义需要检查的物品列表  
    items_to_check = [  
        "AK-47 | Redline (Battle-Scarred)",  
        "AK-47 | Blue Laminate (Field-Tested)",  
        "AK-47 | Asiimov (Factory New)",  
        "AK-47 | Nightwish (Factory New)"  
    ]  

    # 创建检查器并检查价格文件  
    checker = PriceChecker(prices_dir)  
    checker.check_prices(items_to_check)