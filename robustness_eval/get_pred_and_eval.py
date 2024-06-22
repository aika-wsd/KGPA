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


# 将任务多线程化，缩短运行时间键值对列表
def get_pred(loader, td):
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
    # 对dataset的名称进行一些小修改
    args.dataset = args.dataset.lower()
    if args.dataset == 'sst-2':
        args.dataset = 'sst2'

    # test_loader是一个包含处理后的数据和标签的列表
    # label_list：从数据集的元数据中提取一个函数，将整数类型的标签转换为字符串表示
    if (
            args.dataset == 'sst2' or args.dataset == 'qqp' or
            args.dataset == 'mnlt-m' or args.dataset == 'mnlt-mm' or
            args.dataset == 'rte' or args.dataset == 'qnli'
    ):
        test_loader, label_list = get_dataset(args)
    elif (
        args.dataset == 'google_re' or args.dataset == 'trex' or
        args.dataset == 'umls' or args.dataset == 'wiki_bio'
    ):
        test_loader, label_list = kg_get_dataset(args)
    else:
        print('dataset not supported')
        exit(1)

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
    for td_index in tqdm(range(12), desc="Outer Loop"):
        task_description = get_td(td_index, args.dataset)
        pred = get_pred(test_loader, task_description)

        # 初始化存储对抗样本的加载器
        adv_loader = []
        # 内层循环：遍历测试数据加载器中的每个批次
        for batch in tqdm(test_loader, desc="Inner Loop", leave=False):
            # 从批次中提取输入和标签
            batch_x = [x for (x, y) in batch]
            batch_y = [y for (x, y) in batch]
            # 生成对抗样本
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
                task_description=task_description,
            )
            # 将对抗样本及其标签加入对抗样本加载器
            adv_loader.append([[adv_x, y] for (adv_x, y) in zip(batch_adv_x, batch_y)])

        # 获取对抗样本的预测结果
        adv_pred = get_pred(adv_loader, task_description)
        # 计算并存储自然准确率、鲁棒准确率和攻击成功率
        natural_acc.append(get_accuracy(pred, label))
        robust_acc.append(get_accuracy(adv_pred, label))
        ASR.append(get_ASR(pred, adv_pred, label))

        # 打印当前任务描述的性能指标
        print(
            "Task Description Index: {} \t Natural Accuracy: {} Robust Accuracy: {} \t Attack Success Rate: {}".format(
                td_index, natural_acc[td_index], robust_acc[td_index], ASR[td_index]
            )
        )

    # 计算并打印所有任务描述的平均性能指标
    print(
        "Average Natural Accuracy: {} Average Robust Accuracy: {} \t Average Attack Success Rate: {}".format(
            np.mean(natural_acc), np.mean(robust_acc), np.mean(ASR)
        )
    )
