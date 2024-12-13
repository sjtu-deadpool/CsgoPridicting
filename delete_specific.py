import os  
import re

def delete_price_files(directory, items):  
    """  
    删除指定的物品价格文件  
    :param directory: 存放价格文件的目录  
    :param items: 需要删除的物品名称列表  
    """  
    for item_name in items:  
        # 格式化文件名  
        normalized_name = re.sub(r'[\\/*?:"<>|]', '', item_name).replace(' ', '_').replace('|', '-')  
        filename = f"{directory}/{normalized_name}.csv"  

        # 检查文件是否存在，若存在则删除  
        if os.path.exists(filename):  
            os.remove(filename)  
            print(f"✅ 已删除文件: {filename}")  
        else:  
            print(f"❌ 文件不存在: {filename}")  

if __name__ == "__main__":  
    # 定义保存价格文件的目录和需要删除的文件名  
    prices_dir = "item/prices"  # 文件目录  
    items_to_delete = [  
        "AK-47 | Redline (Battle-Scarred)",  
        "AK-47 | Blue Laminate (Field-Tested)",  
        "AK-47 | Asiimov (Factory New)",  
        "AK-47 | Nightwish (Factory New)"  
    ]  

    # 删除指定的文件  
    delete_price_files(prices_dir, items_to_delete)