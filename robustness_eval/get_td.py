# 获取给定数据集和索引对应的任务描述（task description）
# 这段代码在Attack/Attack.py的
import pickle
import ast

def get_td(td_index, args):
    # 从文件对象f中读取序列化的对象
    if args and hasattr(args, 'pkl_file_path'):
        with open(args.pkl_file_path, "rb") as f:
            # 从文件对象f中读取序列化的对象
            td_fsexample_info = pickle.load(f)
            # 如果data是字符串形式的字典
            # print(td_fsexample_info)
            if isinstance(td_fsexample_info, str):
                td_fsexample_info = ast.literal_eval(td_fsexample_info)
    else:
        print("No valid pkl_file_path provided in args.")
    # 从加载的字典中通过键"td"访问存储的任务描述列表，然后使用td_index获取特定索引的任务描述
    td = td_fsexample_info["td"][td_index]
    return td
