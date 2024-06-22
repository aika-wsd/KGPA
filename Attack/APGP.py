# APGP(Attack Prompt Generation Prompt, 攻击性提示词生成提示词模块)
from LLMcalls.LLMCall import LLMCall
import random

class APGP(LLMCall):
    def __init__(
            self,
            dataset,
            label_list,
            args,
    ) -> None:
        super().__init__(args, args.attack_log_file)

        self.dataset = dataset
        if self.dataset == "qqp":
            label_list = ["not_equivalent", "equivalent"]
        # 这里进行了修改，原来给self.label_list赋值是在上面这个判断之前，这会使这个if语句没有用
        self.label_list = label_list

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

    def attack_prompt_generation(
            self, x, y, t_a, perturbation_instruction_index, few_shot_example=None
    ):
        # x是一个列表，包含多个元组，每个元组可能包含一个标记(c_1，如sentence的内容)和其对应的分类(t_1, 如sentence)
        # x = [[t_1,c_1],...,[t_n,c_n]]
        # y是数据实例的真实标签索引
        # y = index of ground-truth label
        # perturbation_instruction_index: 少样本策略
        # few_shot_example: 集成策略

        # 这里就是说，对于每个IDP(Initial Data Point, IDP, 初始是数据点)(x, y)，首先说清楚x
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

        if few_shot_example is not None:
            ag += "Here is an example that fit the guidance: \n"
            # 生成一个在 [0, len(few_shot_example)) 范围内的随机整数
            i = random.randint(0, len(few_shot_example) - 1)
            ag += "{} -> {}\n".format(
                few_shot_example[i][0], few_shot_example[i][1]
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
