from torch.utils.data import Dataset
import random
import json
import os
from LLMcalls.LLMCall import LLMCall
from concurrent.futures import ThreadPoolExecutor

# 将知识图谱的二元组转化为prompt
class T2P(Dataset):
    # 这里args提供了dataset的名字
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.triple_file_path = args.triple_file_path
        # true: True triple
        # entity_error: Wrong Subject or Wrong Object
        # predicate_error: Wrong Predicate
        self.labels = ["true", "entity_error", "predicate_error"]

    def load_predicates(self):
        raise NotImplementedError

        # relation_dict = {}
        # if self.dataset == 'google_re':
        #     # 记录关系的所有字典都应该有类似如下的格式
        #     relation_dict["/people/person/date_of_birth"] = {
        #     "predicate": "date of birth", # 关系本身
        #     "description": "date of birth", # 对该关系的描述
        #  }

    # 从三元组中提取出格式标准的数据集
    # 这里使用了递归，因为需要读取三元组的函数中有可能有文件夹内还有文件夹的情况
    # 这里根据随机选取的标签，可能对三元组的sub、relation、obj中的一个进行修改
    def extract_triples(self, triples=None, triples_file_path=None, predicate_dict=None):
        # 下面是第一层递归的情况
        if triples is None:
            triples = []
        if triples_file_path is None:
            triples_file_path = self.triple_file_path
        if predicate_dict is None:
            predicate_dict = self.load_predicates()
        file_names = os.listdir(triples_file_path)
        # 提取relations_dict中key部分的值的值，便于在后面需要修改relations加以操作
        predicate_dict_keys = list(predicate_dict.keys())
        # print(predicate_dict_keys)

        for file_name in file_names:
            file_path = os.path.join(triples_file_path, file_name)
            # 如果读取到的是一个文件夹，那么继续向里读取，直到读取到文件
            # 在这里使用了递归
            if os.path.isdir(file_path):
                triples = self.extract_triples(
                    triples=triples,
                    triples_file_path=file_path,
                    predicate_dict=predicate_dict
                )
            # 如果读取到的是一个文件，那么从这里面读取对应的三元组
            # 这里根据随机选取的标签，可能对三元组的sub、relation、obj中的一个进行修改
            elif os.path.isfile(file_path) and os.path.splitext(file_name)[0] != 'relations':
                # 暂时存储从文件中解读出来的原始信息（之后从其中解读出三元组）
                data_collection = []
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        original_data = json.loads(line)
                        data_collection.append(original_data)

                # 设置随机数种子
                random.seed(random.randint(0, 666))

                # 为了制造出negative样本，我们从同一类数据中提取出可以替换原三元组中任一部分的元素
                data_collection_length = len(data_collection)
                label_num = len(self.labels)
                for icount in range(data_collection_length):
                    # Extract 'sub' (subject), 'pred' (predicate), and 'obj' (object)
                    # from each JSON line
                    # 读取sub(箭头出发点)与obj(箭头终点)
                    data = data_collection[icount]
                    subs = [data.get("sub_label")] if data.get("sub_label") is not None else []
                    objs = [data.get("obj_label")] if data.get("obj_label") is not None else []

                    obj_labels = data.get("obj_labels", [])
                    objs.extend(obj_labels)

                    # 读取sub与obj的别名
                    # 安全地处理可能不存在的键
                    sub_aliases = data.get("sub_aliases", [])
                    obj_aliases = data.get("obj_aliases", [])

                    # 读取relation
                    predicate_key = data.get("predicate_id", data.get("pred"))
                    predicate = predicate_dict[predicate_key]

                    # label会是["true", "entity_error", "predicate_error"]的一个
                    label = self.labels[random.randint(0, label_num - 1)]
                    # 如果label不是"True"（也就是需要修改原三元组）
                    if not label == "true":
                        # 从此文件中随机选取另一个三元组，用这个三元组的sub或obj替换本三元组的对应部分
                        if label == "entity_error":
                            change_count = random.randint(0, data_collection_length - 1)
                            while change_count == icount:
                                change_count = random.randint(0, data_collection_length - 1)
                            change_data = data_collection[change_count]
                            # 修改subject还是object？
                            sub_or_obj = random.randint(0, 1)
                            if sub_or_obj == 0:
                                subs = [change_data.get("sub_label")] if change_data.get("sub_label") is not None else []
                                sub_aliases = change_data.get("sub_aliases", [])
                            elif sub_or_obj == 1:
                                objs = [change_data.get("obj_label")] if change_data.get("obj_label") is not None else []
                                obj_labels = change_data.get("obj_labels", [])
                                objs.extend(obj_labels)
                                obj_aliases = change_data.get("obj_aliases", [])
                            else:
                                print("Wrong Label")
                                exit(1)
                        elif label == "predicate_error":
                            change_predicate_key = random.choice(predicate_dict_keys)
                            while change_predicate_key == predicate_key:
                                change_predicate_key = random.choice(predicate_dict_keys)
                            predicate = predicate_dict[change_predicate_key]
                        else:
                            print("Wrong Label")
                            exit(1)

                    if label == "true":
                        label = 0
                    elif label == "entity_error":
                        label = 1
                    elif label == "predicate_error":
                        label = 2
                    else:
                        print("Wrong Label")
                        exit(1)

                    triples.append(
                        {
                            "subs": subs,
                            "predicate": predicate,
                            "objs": objs,
                            "sub_aliases": sub_aliases,
                            "obj_aliases": obj_aliases,
                            "label": label,
                        }
                    )

        # 随机生成一个0到666之间的整数作为种子
        seed = random.randint(0, 666)
        random.seed(seed)  # 设置随机数种子

        # 随机打乱列表
        random.shuffle(triples)
        # 截取前1000位（抽样 ）
        triples = triples[:1013]

        return triples

    # 生成prompt
    def template_based_generate_prompts(self):
        raise NotImplementedError

    def llm_based_generate_single_prompt(self, triple):
        subs = triple["subs"]
        predicate = triple["predicate"]
        objs = triple["objs"]
        sub_aliases = triple["sub_aliases"]
        obj_aliases = triple["obj_aliases"]

        # 调用llm
        llm_query = LLMCall(self.args, self.args.t2p_log_file)

        # 设置调用llm的代码，产生描述这个三元组的statement
        prompt_generation_prompt = "Here is a triple (subject, predicate, object) extracted from knowledge graph: \n"
        prompt_generation_prompt += "The subject: {};\n".format(", ".join(subs))
        prompt_generation_prompt += "The predicate: {};\n".format(str(predicate))
        prompt_generation_prompt += "The object: {};\n".format(", ".join(objs))
        prompt_generation_prompt += "Now create a statement describing this triple.\n"
        prompt_generation_prompt += "Tips: Do not care about whether this triple is true, and do not change the meaning of the predicate.\n"
        prompt_generation_prompt += "Just give the statement. Statement:"
        statement = llm_query.query(prompt_generation_prompt)

        # 就是sub relation obj的关系的表述
        prompt = "{}".format(statement)

        # 打印出产生的prompt
        print('Model Version:{}  Dataset:{}  t2p_based_model:{}:\n{}\nLabel:{}\nPredicate:{}\n\n'.format(
            self.args.version,
            self.args.dataset,
            self.args.t2p_based_model,
            prompt,
            triple["label"],
            str(predicate),
        ))

        return prompt, str(subs), str(predicate), str(objs), str(sub_aliases), str(obj_aliases), triple["label"]

    def llm_based_generate_prompts(self):
        # 提取三元组
        triples = self.extract_triples()
        # 利用线程池加速处理
        with ThreadPoolExecutor() as executor:
            generated = list(executor.map(self.llm_based_generate_single_prompt, triples))
        generated_prompts, subs_list, predicates_list, objs_list, sub_aliases_list, obj_aliases_list, labels\
            = zip(*generated)

        return generated_prompts, subs_list, predicates_list, objs_list, sub_aliases_list, obj_aliases_list, labels
