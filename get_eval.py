import os

from tqdm import tqdm
import numpy as np
import concurrent.futures
from LLMcalls.Predict import Predict
from Attack.Attack import Attack
from options import get_args
from robustness_eval.get_dataset import get_dataset
from robustness_eval.get_dataset import kg_get_dataset
from robustness_eval.get_ASR import get_ASR
from robustness_eval.get_td import get_td
from robustness_eval.get_accuracy import get_accuracy
import pickle
import copy
import random
import pandas as pd
import ast


# 将任务多线程化，缩短运行时间键值对列表
def get_pred(loader, td, predictor):
    results = []
    for batch in loader:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # x：键值对列表，td：任务描述
            result = list(
                executor.map(predictor, [x for x, y in batch], [td] * len(batch))
            )
        results += result

    return results

if __name__ == '__main__':
    args = get_args()
    llm_models = ['gpt-3.5-turbo-instruct', 'gpt-4-turbo'] # , 'gpt-4o'] # ['gpt-4-0613']
    kg_datasets = ['trex'] # ['google_re', 'trex', 'umls', 'wiki_bio']
    t2p_based_models = ['template'] # ['llm', 'template']
    tau_llms = [0.70, 0.74, 0.78, 0.82, 0.86, 0.89, 0.92, 0.93, 0.94, 0.96, 0.98, 1.00]

    # 记录llm生成的original prompt与template生成的original prompt之间的区别
    TAU_ASR = [] # 攻击成功率
    TAU_NRA = [] # 自然回答准确率
    TAU_RRA = [] # 对抗成功率
    index = []

    for t2p_based_model in t2p_based_models:
        for llm_model in llm_models:
            args.version = llm_model

            if args.version == 'gpt-3.5-turbo-instruct':
                args.API_base = 'https://api.openai.com/v1'
            else:
                args.API_base = 'https://api.openai.com/v1/chat'


            LLM_T2P_ASR = []
            LLM_T2P_NRA = []
            LLM_T2P_RRA = []
            index.append("{}_{}".format(llm_model, t2p_based_model))

            for kg_dataset in kg_datasets:

                for tau_llm in tau_llms:

                    args.tau_llm = tau_llm

                    args.dataset = kg_dataset
                    args.pkl_file_path = 'info/{}/{}_info.pkl'.format(args.version, args.dataset)

                    original_prompt_file_name = '{}_{}_{}.pkl'.format(llm_model, kg_dataset, t2p_based_model)
                    if t2p_based_model == 'template' and not llm_model == 'gpt-3.5-turbo-instruct':
                        original_prompt_file_name = '{}_{}_{}.pkl'.format(
                           'gpt-3.5-turbo-instruct', kg_dataset, t2p_based_model
                       )
                    print("Original Prompt File Name: {}".format(original_prompt_file_name))
                    original_prompt_file_path = os.path.join('generated_original_prompts', original_prompt_file_name)


                    with open(original_prompt_file_path, 'rb') as f:
                       test_loader_and_label_list = pickle.load(f)
                    # 如果读到的是字符串型数据
                    if isinstance(test_loader_and_label_list, str):
                        test_loader_and_label_list = ast.literal_eval(test_loader_and_label_list)

                    test_loader = copy.deepcopy(test_loader_and_label_list['test_loader'])
                    label_list = copy.deepcopy(test_loader_and_label_list['label_list'])

                    # 随机生成一个0到666之间的整数作为种子
                    seed = random.randint(0, 666)
                    random.seed(seed)  # 设置随机数种子

                    # 随机打乱列表
                    random.shuffle(test_loader)
                    # 截取前13批次（抽样），原因：没米了
                    test_loader = copy.deepcopy(test_loader[:13])

                    # 用于创建一个调用了LLM的预测器
                    predictor = Predict(
                       label_list=label_list,
                       args=args,
                    )

                    # 用于对LLM进行攻击
                    adv_generator = Attack(
                      dataset=args.dataset,
                       label_list=label_list,
                       predictor=predictor,
                        args=args,
                    )

                    # 初始化存储准确率和攻击成功率的列表
                    natural_acc = []
                    robust_acc = []
                    ASR = []

                    # 从测试数据加载器中提取所有标签
                    label = [y for batch in test_loader for x, y in batch]

                    # 说明：我们使用了12种不同的任务描述。这些描述存储在 'info' 文件夹的 pickle 文件中。
                    # 外层循环：遍历12种不同的任务描述
                    # tqdm函数创建了一个进度条
                    # 这里我们只取其中一个当task_description（同样是因为没米了）
                    for td_index in [8]:
                        task_description = get_td(td_index, args)
                        pred = get_pred(test_loader, task_description, predictor)

                        # 初始化存储对抗样本的加载器
                        adv_loader = []
                        # 内层循环：遍历测试数据加载器中的每个批次
                        for batch in tqdm(test_loader, desc="Inner Loop", leave=False):
                            batch_x = [x for (x, y) in batch]
                            batch_y = [y for (x, y) in batch]
                            batch_adv_x = adv_generator.batch_attack(
                               batch_x,
                              batch_y,
                               args.pertub_type,
                               args.t_a,
                               args.tau_wmr,
                               args.tau_bert,
                              args.tau_llm,
                               few_shot=args.few_shot,
                               ensemble=args.ensemble,
                               task_description=task_description,)
                            # 将对抗样本及其标签加入对抗样本加载器
                            adv_loader.append([[adv_x, y] for (adv_x, y) in zip(batch_adv_x, batch_y)])

                        # 获取对抗样本的预测结果
                        adv_pred = get_pred(adv_loader, task_description, predictor)
                        # 计算并存储自然准确率、鲁棒准确率和攻击成功率
                        natural_acc.append(get_accuracy(pred, label))
                        robust_acc.append(get_accuracy(adv_pred, label))
                        ASR.append(get_ASR(pred, adv_pred, label))

                        # 打印当前任务描述的性能指标
                        print("args.tau_llm = {}".format(args.tau_llm))
                        print(
                            "Task Description Index: {} \t Natural Accuracy: {} Robust Accuracy: {} \t \nAttack Success Rate: {}".format(
                                td_index, natural_acc[-1], robust_acc[-1], ASR[-1]
                           )
                        )

                    # 计算并打印所有任务描述的平均性能指标
                    print("args.tau_llm = {}".format(args.tau_llm))
                    print(
                        "Average Natural Accuracy: {} Average Robust Accuracy: {} \t \nAverage Attack Success Rate: {} \n\n\n".format(
                            np.mean(natural_acc), np.mean(robust_acc), np.mean(ASR)
                        )
                    )

                    LLM_T2P_ASR.append(np.mean(ASR))
                    LLM_T2P_NRA.append(np.mean(natural_acc))
                    LLM_T2P_RRA.append(np.mean(robust_acc))

                TAU_ASR.append(copy.deepcopy(LLM_T2P_ASR))
                TAU_NRA.append(copy.deepcopy(LLM_T2P_NRA))
                TAU_RRA.append(copy.deepcopy(LLM_T2P_RRA))

        columns = [0.70, 0.74, 0.78, 0.82, 0.86, 0.89, 0.92, 0.93, 0.94, 0.96, 0.98, 1.00]
        TAU_ASR = pd.DataFrame(TAU_ASR, columns=columns, index=index)
        TAU_NRA = pd.DataFrame(TAU_NRA, columns=columns, index=index)
        TAU_RRA = pd.DataFrame(TAU_RRA, columns=columns, index=index)

        TAU_ASR.to_excel('NO_FS_ASR.xlsx', index=True, header=True)
        TAU_NRA.to_excel('NO_FS_NRA.xlsx', index=True, header=True)
        TAU_RRA.to_excel('NO_FS_RRA.xlsx', index=True, header=True)
