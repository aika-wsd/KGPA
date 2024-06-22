# 加载数据集
from datasets import load_dataset
from Dataset.google_re_T2P import GoogleReT2P
from Dataset.trex_T2P import TrexT2P
from Dataset.umls_T2P import UmlsT2P
from Dataset.wiki_bio_T2P import WikiBioT2P

def get_dataset(args):
    # 检查传入的数据集名称中是否包含"mnli"
    # MNLI（Multi-Genre Natural Language Inference）数据集是一个广泛用于自然语言理解（NLU）的数据集
    # 专门用于文本蕴含（也称为自然语言推理）任务
    # 这个任务的目标是判断一对句子之间的关系，即一个句子（前提）是否蕴含、矛盾或与另一个句子（假设）无关
    dataset = args.dataset
    if "mnli" in dataset:
        assert "mnli-m" in dataset
        if dataset == "mnli-m":
            # 加载GLUE数据集中的MNLI matched validation数据集
            # 在这个子集中，前提和假设句子来自于同一体裁
            dataset_ = load_dataset("glue", "mnli", split="validation_matched")
        elif dataset == "mnli-mm":
            # 加载GLUE数据集中的MNLI mismatched validation数据集
            # 在这个子集中，前提和假设句子来自不同体裁
            # 前提可能是来自法律文件的句子，而假设则来自小说
            dataset_ = load_dataset("glue", "mnli", split="validation_mismatched")
        else:
            raise ValueError("Unknown dataset")
    else:
        # 如果不加载mnli数据集，则
        dataset_ = load_dataset("glue", dataset, split="validation")

    # 这个列表推导式创建了test_loader，它构建了一个包含处理后的数据和标签的列表
    # 对于数据集中的每一行，它构建一个包含所有除"label"和"idx"外的键值对的列表，并将其与该行的"label"组合在一起
    # 这段代码也见于Dataset/CustomDataset.py
    test_loader = [
        (
            [
                [key, value]
                for (key, value) in dataset_[i].items()
                if key != "label" and key != "idx"
            ],
            dataset_[i]["label"],
        )
        for i in range(dataset_.num_rows)
    ]
    # 将test_loader分成多个批次，每个批次的大小由args.batch_size4指定
    test_loader = [
        test_loader[i : i + args.batch_size]
        for i in range(0, len(test_loader), args.batch_size)
    ]
    # 从数据集的元数据中提取一个函数，将整数类型的标签转换为字符串表示
    label_list = dataset_.features["label"]._int2str
    return test_loader, label_list

# 适用于知识图谱的test_loader与label_list提取
def kg_get_dataset(args):
    dataset = args.dataset
    if dataset == "google_re":
        dataset_ = GoogleReT2P(args)
    elif dataset == "trex":
        dataset_ = TrexT2P(args)
    elif dataset == "umls":
        dataset_ = UmlsT2P(args)
    elif dataset == "wiki_bio":
        dataset_ = WikiBioT2P(args)
    else:
        raise ValueError("Unknown dataset")

    # 这个列表推导式创建了test_loader，它构建了一个包含处理后的数据和标签的列表
    # 对于数据集中的每一行，它构建一个包含所有除"label"和"idx"外的键值对的列表，并将其与该行的"label"组合在一起
    # 这段代码也见于Dataset/CustomDataset.py
    test_loader = [
        (
            [
                ["sentence", prompt],
                # ["subject", sub],
                # ["predicate", predicate],
                # ["object", obj],
                # ["subject_aliases", sub_aliases],
                # ["object_aliases", obj_aliases],
            ],
            label,
        )
        for prompt, sub, predicate, obj, sub_aliases, obj_aliases, label in dataset_
    ]
    # 将test_loader分成多个批次，每个批次的大小由args.batch_size4指定
    test_loader = [
        test_loader[i : i + args.batch_size]
        for i in range(0, len(test_loader), args.batch_size)
    ]

    return test_loader, dataset_.labels
