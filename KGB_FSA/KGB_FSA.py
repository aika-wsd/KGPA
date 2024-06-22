import copy
from LLMcalls.LLMCall import LLMCall
from robustness_eval.get_dataset import kg_get_dataset
from concurrent.futures import ThreadPoolExecutor
import random
import pickle
import os
from PRE.PRE import PRE

# 专用于KGPA，生成用于知识图谱数据集的少样本示例
# 另外，这里顺便也写了一个task_description函数，用于生成info文件夹中{}_info.pkl类型的文件
class KGB_FSA(LLMCall):
    def __init__(
            self,
            label_list,
            args,
    ) -> None:
        super().__init__(args, args.kgb_fsa_log_file)
        self.label_list = label_list
        self.args = args

        # 用于在后面的AG(Attack Guidance, AG, 攻击指示)中，提供可选择的攻击指示
        self.perturbation_instruction = [
            "Choose at most two words in the sentence, and change them so that they have typos.",
            "Change at most two letters in the sentence.",
            "Add at most two extraneous characters to the end of the sentence.",
            "Replace at most two words in the sentence with synonyms.",
            "Choose at most two words in the sentence that do not contribute to the meaning of the sentence and delete them.",
            "Add at most two semantically neutral words to the sentence.",
            "Add a randomly generated short meaningless handle after the entence, such as @fasuv3.",
            "Paraphrase the sentence.",
            "Change the syntactic structure of the sentence.",
        ]

        self.task_descriptions = [
            "Analyze the category of the sentence extracted from a triple (subject, predict, object) and respond with 'true' or 'entity_error' or 'predicate_error':\n",
            "Evaluate the category of a sentence derived from a triple (subject, predicate, object) and respond with either 'true', 'entity_error' or 'predicate_error':\n",
            "Assess the type of the sentence taken from a triple (subject, predicate, object) and provide a response of 'true', 'entity_error' or 'predicate_error':\n",
            "Analyze the classification of a sentence from a triple (subject, predicate, object) and reply with 'true', 'entity_error' or 'predicate_error':\n",
            "Determine the category of a sentence sourced from a triple (subject, predicate, object) and answer with 'true', 'entity_error' or 'predicate_error':\n",
            "Examine the type of the sentence extracted from a triple (subject, predicate, object) and respond with 'true', 'entity_error' or 'predicate_error':\n",
            "Review the category of a sentence from a triple (subject, predicate, object) and return a response of 'true', 'entity_error' or 'predicate_error':\n",
            "Classify a sentence taken from a triple (subject, predicate, object) and indicate whether it is 'true', 'entity_error' or 'predicate_error':\n",
            "Appraise the category of a sentence derived from a triple (subject, predicate, object) and decide if it is 'true', 'entity_error' or 'predicate_error':\n",
            "Analyze the classification of a sentence that comes from a triple (subject, predicate, object) and provide a verdict of 'true', 'entity_error' or 'predicate_error':\n",
            "Investigate the category of a sentence based on a triple (subject, predicate, object) and give an answer of 'true', 'entity_error' or 'predicate_error':\n",
            "Identify the category of a sentence originating from a triple (subject, predicate, object) and respond accordingly with 'true', 'entity_error' or 'predicate_error':\n",
        ]

    # 用于检测生成的few_shot_example会让llm产生什么结果
    def few_shots_examples_attack_result(self, x):
        choose_prompt = "Here is a sentence generated from a triplet (subject, predicate, object),"
        choose_prompt += "Classify the correctness of the generated sentence as"
        choose_prompt += "either 'true', 'entity_error' or 'predicate_error'."
        choose_prompt += "Respond with 'true' or 'entity_error' or 'predicate_error': \n"
        type_num = len(x)
        for i in range(type_num):
            choose_prompt += '{} "{}" '.format(x[i][0], x[i][1])
            if i != type_num - 1:
                choose_prompt += "and \n"
        choose_prompt += "\nAnswer: "

        choose_answer = self.query(choose_prompt)

        # 我们的label_list是这样的["true", "entity_error", "predicate_error"]
        # 取_前的第一个单词就能得到合适的结果
        # 对一个字符串choose_answer中的特定标签进行统计分析
        # 确定哪些标签的纯出现次数最高，并找出这些次数最高的标签索引

        # 计算每个标签的纯出现次数（即减去以特定字符_开头的同标签次数）并生成一个列表counts
        counts = [
            choose_answer.count(label.split('_')[0]) for label in self.label_list
        ]
        # 从counts列表中找出最大值，存储在变量max_value中
        max_value = max(counts)
        # 找出所有counts列表中值等于max_value的索引，并将这些索引存储在列表max_indices中
        max_indices = [i for i, value in enumerate(counts) if value == max_value]

        return max_indices[0] if len(max_indices) == 1 else None

    # 检测少样本能否攻击成功
    def few_shots_examples_is_success_attack(self, x, y):
        return self.few_shots_examples_attack_result(x) != y

    def few_shot_examples_generation(
            self, x, y, t_a, perturbation_instruction_index,
    ):
        # x是一个列表，包含多个元组，每个元组可能包含一个标记(c_1，如sentence的内容)和其对应的分类(t_1, 如sentence)
        # x = [[t_1,c_1],...,[t_n,c_n]]
        # y是数据实例的真实标签索引
        # y = index of ground-truth label
        # perturbation_instruction_index: 少样本策略
        # few_shot_example: 集成策略

        # 这里就是说，对于每个IDP(Initial Data Point, IDP, 初始是数据点)(x, y)，首先说清楚x
        # 此处的label_list是
        type_num = len(x)
        idp = "The original"
        for i in range(type_num):
            idp += '{} "{}" '.format(x[i][0], x[i][1])
            if i != type_num - 1:
                idp += "and "
        # 再加上IDP(x, y)的标签y
        idp += "is classified as {}. \n".format(self.label_list[y])

        # 设置ATC(Attack Target Core, ATC, 攻击靶心)
        # 首先指定修改x当中第t_a条
        # 1. 要求保持语义不变
        # 2. 要求能够使得被修改后的IDP的y值发生变动
        atc = "Your task is to generate a new {} which must satisfy the following conditions: \n".format(
            x[t_a][0]
        )
        atc += (
            "1. Keeping the semantic meaning of the new {} unchanged; \n".format(
                x[t_a][0]
            )
        )
        atc += "2. The new {} ".format(x[t_a][0])
        if type_num > 1:
            for i in range(type_num):
                if i != t_a:
                    atc += " and the original {}, ".format(x[i][0])
        atc += "should be classified as "
        for i in range(len(self.label_list)):
            if i != y:
                atc += "{} ".format(self.label_list[i])
                if i != len(self.label_list) - 2:
                    atc += "or "
        atc += ". \n"

        # 设置AG(Attack Guidance, AG, 攻击指示)
        # 首先从self.perturbation_instruction中选择合适的攻击指示
        # 之后，如果few_shot_example被设置为None,则不使用少样本策略
        # 否则，使用少样本策略（即向llm提供一部分示例）
        ag = "You can finish the task by modifying {} using the following guidance: \n".format(
            x[t_a][0]
        )
        ag += "{} \n".format(
            self.perturbation_instruction[perturbation_instruction_index]
        )
        ag += "Only output the new {} without anything else.".format(
            x[t_a][0]
        )

        # AGS(Attack Guidance System, 攻击导向系统)
        ags = atc + ag
        # APGP(Attack Prompt Generation Prompt, APGP, 攻击性提示词生成提示词模块)
        apgp = idp + ags + "\n"
        apgp = apgp + "{} ->".format(x[t_a][1])

        return apgp

    def special_perturbation_instruction_few_shot_examples_generation(
            self, t_a, perturbation_instruction_index,
    ):
        # PRE在此处的应用
        pre = PRE(self.args)

        generated_few_shot_examples = []

        test_loader, _ = kg_get_dataset(self.args)
        max_example_num = 5
        now_example_num = 0
        # 设置随机数种子
        random.seed(random.randint(0, 666))

        while_round = 0
        while now_example_num < max_example_num and while_round < 80:
            while_round += 1

            # 随机选择一个数据点制作少样本数据
            batch_num = len(test_loader)
            single_data_num_in_batch = len(test_loader[-1])
            batch_no = random.randint(0, batch_num - 1)
            single_data_no = random.randint(0, single_data_num_in_batch - 1)
            original_values, label_now = test_loader[batch_no][single_data_no]
            values_now = copy.deepcopy(original_values)
            apgp = self.few_shot_examples_generation(
                values_now, label_now, t_a, perturbation_instruction_index
            )
            # 避免轮询次数过多
            # time.sleep(2)
            values_now[t_a][1] = self.query(apgp)
            # 避免轮询次数过多
            # time.sleep(2)
            # 需要检测是否攻击成功
            # 使用pre.fidelity_filter是确保得到的提示词是最优的
            filtered_sample, _, _, _ = pre.fidelity_filter(
                copy.deepcopy(original_values[t_a][1]),
                copy.deepcopy(values_now[t_a][1]),
                self.args.tau_wmr, self.args.tau_bert, self.args.tau_llm
            )
            # print(filtered_sample)
            # print(original_values[t_a][1])
            if (
                    self.few_shots_examples_is_success_attack(values_now, label_now) and
                    not (original_values[t_a][1] == values_now[t_a][1]) and
                    not (filtered_sample == original_values[t_a][1])
            ):

                # 测试时用
                print(original_values[t_a][1])
                print(values_now[t_a][1])
                print('\n')

                generated_few_shot_examples.append([original_values[t_a][1], values_now[t_a][1]])
                now_example_num += 1

        return generated_few_shot_examples

    # 生成info文件夹中的文件
    def generate_info_file(self):
        info_file_data = {}

        # 利用线程池加速处理
        with ThreadPoolExecutor() as executor:
            generated = list(executor.map(
                self.special_perturbation_instruction_few_shot_examples_generation,
                [0] * len(self.perturbation_instruction),
                list(range(len(self.perturbation_instruction))),
            ))

        info_file_data['fs_example'] = generated
        info_file_data['td'] = copy.deepcopy(self.task_descriptions)

        file_path = os.path.join('info', '{}_info.pkl'.format(self.args.dataset))
        with open(file_path, 'wb') as file:
            pickle.dump(info_file_data, file)

        return
