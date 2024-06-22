# PRE(Prompt Generation Engine, 提示词优化模块)
from LLMcalls.LLMCall import LLMCall

from concurrent.futures import ThreadPoolExecutor
import nltk
from bert_score import score
from nltk.tokenize import word_tokenize
import itertools

# 匹配str中出现的小数或数字
import re

# nltk.download("punkt")这行代码的作用是从NLTK（自然语言处理工具包）的服务器下载punkt数据包
# punkt是一个用于句子分界的预训练模型，可以自动分割文本成句子，这主要用于将文本分割成单独的句子
# 这是进行诸如词频分析、情感分析等更复杂自然语言处理任务的初步步骤。
# nltk.download("punkt")

class PRE:
    def __init__(
            self,
            args,
    ) -> None:
        self.args = args

    # 使用动态规划算法计算sentence1与sentence2之间的Levenshtein distance
    # 然后再与sentence1（被修改的句子）的单词数相除，得到单词修改率
    def get_word_modification_ratio(self, sentence1, sentence2):
        return 0
        """
        # 将sentence1和sentence2进行分词，形成两个词列表
        words1, words2 = word_tokenize(sentence1), word_tokenize(sentence2)
        m, n = len(words1), len(words2)
        # 初始化一个(m + 1)*(n + 1)的二维列表dp
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        # 初始化动态规划表的第一列，其中dp[i][0]的值是从words1的第一个词到第i个词的删除操作次数
        for i in range(m + 1):
            dp[i][0] = i
        # 初始化动态规划表的第一行，其中dp[0][j]的值是从words2的第一个词到第j个词的插入操作次数
        for j in range(n + 1):
            dp[0][j] = j
        # 使用itertools.product生成所有可能的(i, j)组合，这些组合代表dp表中的位置，从1开始到m和n
        for i, j in itertools.product(range(1, m + 1), range(1, n + 1)):
            # 计算替换成本cost
            # 如果words1的第i-1个词和words2的第j-1个词相同，则无需替换（成本为 0），否则替换成本为1
            cost = 0 if words1[i - 1] == words2[j - 1] else 1
            # 计算到达dp[i][j]的最小编辑距离
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        # dp[m][n] / m就是单词修改率，即被修改的单词占原始单词总数的比例
        return dp[m][n] / m
        """

    def llm_score(self, sentence1, sentence2):
        # 设计用于给出llm_score的llm评分器
        llm_query = LLMCall(self.args, self.args.pre_log_file)
        # compare_prompt
        compare_prompt = "Task Description: \n"
        compare_prompt += ("you are required to "
                           "assess the quality of a piece of generated text, "
                           "as well as its semantic similarity and overall quality "
                           "in relation to a provided reference text. \n")
        compare_prompt += "Input Data: \n"
        compare_prompt += "Reference Text: {} \n".format(sentence1)
        compare_prompt += "Candidate Text: {} \n".format(sentence2)
        compare_prompt += ("Scoring Guidelines: Please rate the candidate text according to "
                           "the following criteria, "
                           "with scores ranging from -1 to 1, "
                           "where 1 represents a perfect match, "
                           "and -1 represents a complete mismatch. "
                           "The score can be a decimal\n")
        compare_prompt += "Just give the score, and the Score needs to be a three-digit decimal representation, score:"
        # llm的回答
        llm_score_answer = llm_query.query(compare_prompt)

        match = re.search(r'[-+]?\d+\.?\d*', llm_score_answer)
        if match:
            # 如果找到了匹配的数字，返回匹配的部分
            match_num =  float(match.group()) if '.' in match.group() else int(match.group())
            # print(match_num)
            if match_num < -1:
                match_num = -1
            elif match_num > 1:
                match_num = 1
            return match_num
        else:
            # print(-1)
            return -1

    # 比较原始样本（ori_sample）和修改后的样本（adv_sample）
    # 根据给定的阈值（tau_wmr和tau_bert）决定是否接受修改
    # tau_wmr是单词修改率（word modification ratio）的阈值，tau_bert是BERTScore的阈值
    # 预计还会添加LLM自评分（LLMScore）的阈值tau_llm
    def fidelity_filter(self, ori_sample, adv_sample, tau_wmr, tau_bert, tau_llm):
        # 直接输出修改前后的样本，便于观察
        # print("{} -> {} \n".format(ori_sample, adv_sample))
        # 得到单词修改率
        wmr = self.get_word_modification_ratio(ori_sample, adv_sample)
        # 得到BERTScore
        # lang="en"指定了使用的 BERT 模型的语言版本，表示使用英语模型
        # Precision与Recall被省略，只取F1-score为BERTScore
        # _, _, BERTScore = score([ori_sample], [adv_sample], lang="en")
        # BERTScore = BERTScore[0].item()
        # 通过llm得到LLMScore
        LLMScore = self.llm_score(ori_sample, adv_sample)

        # 如果wmr与BERTScore都在阈值允许范围内，则接受修改之后的sentence
        # 否则输出原sentence
        # if wmr <= tau_wmr and BERTScore >= tau_bert and LLMScore >= tau_llm:
        if wmr <= tau_wmr and LLMScore >= tau_llm:
            # return adv_sample, wmr, BERTScore
            print("1: {} -> {}".format(ori_sample, adv_sample))
            return adv_sample, wmr, 0, LLMScore
        # return ori_sample, wmr, BERTScore, LLMScore
        print("0: {} -> {}".format(ori_sample, adv_sample))
        return ori_sample, wmr, 0, LLMScore

    # 下面这个函数利用上面的fidelity_filter函数批量处理文本
    # 根据给定的阈值（tau_wmr和tau_bert）决定是否接受修改
    # tau_wmr是单词修改率（word modification ratio）的阈值，tau_bert是BERTScore的阈值
    # 预计还会添加LLM自评分（LLMScore）的阈值tau_llm
    def batch_fidelity_filter(self, ori_samples, adv_samples, tau_wmr, tau_bert, tau_llm):
        # 得到单词修改率
        # zip将ori_samples与adv_samples中对应位置的列表配对
        wmrs = [
            self.get_word_modification_ratio(ori_sample, adv_sample)
            for (ori_sample, adv_sample) in zip(ori_samples, adv_samples)
        ]
        # 得到BERTScore
        # 使用的是和fidelity_filter中一样的函数
        # Precision与Recall被省略，只取F1-score为BERTScore
        # 在BERTScores = BERTScores.tolist()之前，BERTScore是一个张量
        # _, _, BERTScores = score(ori_samples, adv_samples, lang="en")
        # BERTScores = BERTScores.tolist()
        # 批量得到LLMScore
        # 利用线程池加速处理
        with ThreadPoolExecutor() as executor:
            LLMScores = list(executor.map(self.llm_score, ori_samples, adv_samples))

        # 如果wmr与BERTScore都在阈值允许范围内，则接受修改之后的sentence
        # 否则输出原sentence
        results = [
            adv_sample
            # if wmr <= tau_wmr and BERTScore >= tau_bert and LLMScore >= tau_llm
            if wmr <= tau_wmr and LLMScore >= tau_llm
            else ori_sample
            for (ori_sample, adv_sample, wmr, BERTScore, LLMScore) in zip(
                # ori_samples, adv_samples, wmrs, BERTScores, LLMScores
                ori_samples, adv_samples, wmrs, [1] * len(wmrs), LLMScores
            )
        ]

        return results, wmrs, [1] * len(wmrs), LLMScores
