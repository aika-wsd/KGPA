import copy
from Attack.APGP import APGP
from PRE.PRE import PRE
import pickle
import concurrent.futures
import random

class Attack(APGP):
    def __init__(
            self,
            dataset,
            label_list,
            predictor,
            args,
    ) -> None:
        super().__init__(
            dataset,
            label_list,
            args,
        )
        # 保存传入的预测模型
        self.predictor = predictor
        self.args_for_PRE = args

        """
        # 以二进制读形式打开路径info/{self.dataset}_info.pkl文件
        # 这里运行的时候要注意一下能否正常载入
        with open(os.path.join("info", "{}_info.pkl".format(self.dataset)), "rb") as info:
            # 从文件对象f中读取序列化的对象
            self.td_fsexample_info = pickle.load(info)
        """

        # 这里我们选择给文件路径选择更大的自由
        if args and hasattr(args, 'pkl_file_path'):
            with open(args.pkl_file_path, "rb") as f:
                # 从文件对象f中读取序列化的对象
                self.td_fsexample_info = pickle.load(f)
        else:
            print("No valid pkl_file_path provided in args.")

    # 少样本策略的一部分，从数据集中提取出一个，作为给llm的提示使用
    def get_fewshot_examples(self, perturbation_instruction_index):
        return self.td_fsexample_info["fs_example"][perturbation_instruction_index]

    # 如果修改后的prompt使得LLM输出了错误的预测结果，那么认为此次攻击成功
    def is_success_attack(self, x, y, task_description):
        return self.predictor(x, task_description) != y

    def attack(
        self,
        x, # 输入样本
        y, # 样本标签
        perturbation_instruction_index, # 用于选择具体扰动指令的索引
        t_a, # 目标样本的索引，指定修改x当中第t_a条
        # 实际上tau_wmr的值在函数中被修改，这里没有任何影响
        tau_wmr, # 单词修改率的阈值
        tau_bert, # BERTScore的阈值
        tau_llm, # LLMScore的阈值
        few_shot=False, # 是否使用少样本策略
        ensemble=False, # 是否使用集成策略
        task_description=None, # 针对集成方法，需要提供的任务描述
    ):
        # 随机生成一个0到666之间的整数作为种子
        seed = random.randint(0, 666)
        random.seed(seed)  # 设置随机数种子

        assert 0 <= tau_wmr and tau_wmr <= 1, "wmr must be in range [0, 1]"
        assert 0 <= tau_bert and tau_bert <= 1, "bert must be in range [0, 1]"
        assert -1 <= tau_llm and tau_llm <= 1, "llm must be in range [-1, 1]"
        # 在使用集成策略时，需要确保predictor需要被正确设置
        if ensemble:
            assert self.predictor is not None

        # 如果启用了few-shot学习，根据提供的索引从存储的信息中获取相应的few-shot示例；
        # 如果未启用，则将few_shot_example设置为None
        if few_shot:
            few_shot_example = self.get_fewshot_examples(perturbation_instruction_index)
        else:
            few_shot_example = None

        # 使用PRE模块生成提示词优化器
        pre = PRE(self.args_for_PRE)

        # 没使用集成策略的情况，执行下列代码块
        if not ensemble:
            # 调用APGP模块生成apgp(attack prompt generation prompt)
            apgp = self.attack_prompt_generation(
                x, y, t_a, perturbation_instruction_index, few_shot_example
            )
            # 将apgp输入llm中生成攻击性样本
            adv_sample = self.query(apgp)
            if self.dataset == "sst2":
                # 如果使用sst2数据集中的数据，则将adv_sample全部转为小写
                adv_sample = adv_sample.lower()
            # 根据扰动指令的索引调整tau_1的值。
            # 如果索引大于等于7，扰动幅度不受限制（设为 1.0）
            # 否则，设置为0.15，限制扰动程度
            # 这里会导致传入的tau_wmr值没有任何作用
            tau_wmr = 1.0 if perturbation_instruction_index >= 7 else 0.15
            # 检测一下adv_sample在tau_wmr与tau_bert与tau_llm的阈值上能否过关
            # 如果合格，就使用adv_sample，否则使用原来的样本
            adv_sample, _, _, _ = pre.fidelity_filter(x[t_a][1], adv_sample, tau_wmr, tau_bert, tau_llm)
            # 深拷贝输入样本x，并用接受的修改后样本更新x中t_a的位置
            adv_x = copy.deepcopy(x)
            adv_x[t_a][1] = adv_sample

        else:
            assert task_description is not None
            adv_x = copy.deepcopy(x)
            BERTScore = 0.0
            LLMScore = 0.0
            for i in range(len(self.perturbation_instruction)):
                # 调用APGP模块生成apgp(attack prompt generation prompt)
                apgp = self.attack_prompt_generation(x, y, t_a, i, few_shot_example)
                # 将apgp输入llm中生成攻击性样本
                adv_sample = self.query(apgp)
                if self.dataset == "sst2":
                    # 如果使用sst2数据集中的数据，则将adv_sample全部转为小写
                    adv_sample = adv_sample.lower()
                # 根据扰动指令的索引调整tau_1的值。
                # 如果索引大于等于7，扰动幅度不受限制（设为 1.0）
                # 否则，设置为0.15，限制扰动程度
                # 这里会导致传入的tau_wmr值没有任何作用
                tau_wmr = 1.0 if perturbation_instruction_index >= 7 else 0.15
                # 检测一下adv_sample在tau_wmr与tau_bert与tau_llm的阈值上能否过关
                # 如果合格，就使用adv_sample，否则使用原来的样本
                adv_sample, _, tmp_bertscore, tmp_llmscore = pre.fidelity_filter(
                    x[t_a][1], adv_sample, tau_wmr, tau_bert, tau_llm
                )
                tmp_adv_x = copy.deepcopy(x)
                tmp_adv_x[t_a][1] = adv_sample
                # 下面的if语句的目的是，从perturbation_instruction的列表中选择效果最好的一个
                if (
                    # 首先要求能够成功攻击
                    self.is_success_attack(tmp_adv_x, y, task_description)
                    # 同时要求达到最大的BERTScore与LLMScore值
                    and tmp_bertscore > BERTScore and tmp_llmscore > LLMScore
                ):
                    adv_x = tmp_adv_x
                    BERTScore = tmp_bertscore
                    LLMScore = tmp_llmscore
            return adv_x

        # adv_x是生成的攻击性提示词
        return adv_x

    def batch_attack(
        self,
        batch_x, # 输入样本
        batch_y, # 样本标签
        perturbation_instruction_index, # 用于选择具体扰动指令的索引
        t_a, # 目标样本的索引，指定修改x当中第t_a条
        tau_wmr, # 单词修改率的阈值
        tau_bert, # BERTScore的阈值
        tau_llm, # LLMScore的阈值
        few_shot=False, # 是否使用少样本策略
        ensemble=False, # 是否使用集成策略
        task_description=None, # 针对集成方法，需要提供的任务描述
    ):
        # 随机生成一个0到666之间的整数作为种子
        seed = random.randint(0, 666)
        random.seed(seed)  # 设置随机数种子

        assert 0 <= tau_wmr and tau_wmr <= 1, "wmr must be in range [0, 1]"
        assert 0 <= tau_bert and tau_bert <= 1, "bert must be in range [0, 1]"
        assert 0 <= tau_llm and tau_llm <= 1, "llm must be in range [0, 1]"
        # 在使用集成策略时，需要确保predictor需要被正确设置
        if ensemble:
            assert self.predictor is not None

        # 如果启用了few-shot学习，根据提供的索引从存储的信息中获取相应的few-shot示例；
        # 如果未启用，则将few_shot_example设置为None
        if few_shot:
            few_shot_example = self.get_fewshot_examples(perturbation_instruction_index)
        else:
            few_shot_example = None

        # 使用PRE模块生成提示词优化器
        pre = PRE(self.args_for_PRE)

        # 没使用集成策略的情况，执行下列代码块
        # concurrent.futures.ThreadPoolExecutor(): 这是一个执行器（executor）对象，用于创建和管理线程池
        # ThreadPoolExecutor是基于线程的执行器，适用于执行多个任务的场景
        # 尤其是任务本身不是密集计算型的，而可能涉及等待（如网络请求、文件读写等)
        # 用executor来提交任务到线程池
        if not ensemble:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                apgp = list(
                    executor.map(
                        self.attack_prompt_generation,
                        batch_x,
                        batch_y,
                        [t_a] * len(batch_x),
                        [perturbation_instruction_index] * len(batch_x),
                        [few_shot_example] * len(batch_x),
                    )
                )
            # 将apgp输入llm中生成攻击性样本
            # 同样使用了线程池的技术
            with concurrent.futures.ThreadPoolExecutor() as executor:
                adv_samples = list(executor.map(self.query, apgp))
            if self.dataset == "sst2":
                # 如果使用sst2数据集中的数据，则将adv_sample全部转为小写
                adv_samples = [adv_sample.lower() for adv_sample in adv_samples]
            # 根据扰动指令的索引调整tau_1的值。
            # 如果索引大于等于7，扰动幅度不受限制（设为 1.0）
            # 否则，设置为0.15，限制扰动程度
            # 这里会导致传入的tau_wmr值没有任何作用
            tau_wmr = 1.0 if perturbation_instruction_index >= 7 else 0.15
            # 检测一下adv_sample在tau_wmr与tau_bert的阈值上能否过关
            # 如果合格，就使用adv_sample，否则使用原来的样本
            adv_samples, _, _, _ = pre.batch_fidelity_filter(
                [x[t_a][1] for x in batch_x], adv_samples, tau_wmr, tau_bert, tau_llm
            )
            # 深拷贝输入样本x，并用接受的修改后样本更新x中t_a的位置
            batch_adv_x = copy.deepcopy(batch_x)
            for adv_x, adv_sample in zip(batch_adv_x, adv_samples):
                adv_x[t_a][1] = adv_sample

        else:
            assert task_description is not None
            bertscores = [0.0 for i in range(len(batch_x))]
            llmscores = [0.0 for i in range(len(batch_x))]
            batch_adv_x = copy.deepcopy(batch_x)
            for i in range(len(self.perturbation_instruction)):
                # 调用APGP模块生成apgp(attack prompt generation prompt)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    apgp = list(
                        executor.map(
                            self.attack_prompt_generation,
                            batch_x,
                            batch_y,
                            [t_a] * len(batch_x),
                            [i] * len(batch_x),
                            [few_shot_example] * len(batch_x),
                        )
                    )
                # 将apgp输入llm中生成攻击性样本
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    adv_samples = list(executor.map(self.query, apgp))
                # 如果使用sst2数据集中的数据，则将adv_sample全部转为小写
                if self.dataset == "sst2":
                    adv_samples = [adv_sample.lower() for adv_sample in adv_samples]
                # 根据扰动指令的索引调整tau_1的值。
                # 如果索引大于等于7，扰动幅度不受限制（设为 1.0）
                # 否则，设置为0.15，限制扰动程度
                # 这里会导致传入的tau_wmr值没有任何作用
                tau_wmr = 1.0 if i >= 7 else 0.15
                # 检测一下adv_sample在tau_wmr与tau_bert的阈值上能否过关
                # 如果合格，就使用adv_sample，否则使用原来的样本
                adv_samples, _, tmp_bertscores, tmp_llmscores = pre.batch_fidelity_filter(
                    [x[t_a][1] for x in batch_x], adv_samples, tau_wmr, tau_bert, tau_llm
                )
                batch_tmp_adv_x = copy.deepcopy(batch_x)

                for tmp_adv_x, adv_sample in zip(batch_tmp_adv_x, adv_samples):
                    tmp_adv_x[t_a][1] = adv_sample

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    success_attacks = list(
                        executor.map(
                            self.is_success_attack,
                            batch_tmp_adv_x,
                            batch_y,
                            [task_description] * len(batch_tmp_adv_x),
                        )
                    )

                for (
                    i,
                    (
                        adv_x,
                        tmp_adv_x,
                        tmp_bertscore,
                        bertscore,
                        tmp_llmscore,
                        llmscore,
                        success_attack,
                    )
                ) in enumerate(
                    zip(
                        batch_adv_x,
                        batch_tmp_adv_x,
                        tmp_bertscores,
                        bertscores,
                        tmp_llmscores,
                        llmscores,
                        success_attacks,
                    )
                ):
                    # 下面的if语句的目的是，从perturbation_instruction的列表中选择效果最好的一个
                    if success_attack and tmp_bertscore > bertscore and tmp_llmscore > llmscore:
                        # 这里对原代码进行了改正，个人认为改对了
                        batch_adv_x[i] = tmp_adv_x
                        bertscores[i] = tmp_bertscore
                        llmscores[i] = tmp_llmscore

        # # adv_x是生成的攻击性提示词
        return batch_adv_x
