from LLMcalls.LLMCall import LLMCall

class Predict(LLMCall):
    def __init__(
            self, label_list, args
    ) -> None: # REVISE
        super().__init__(args, args.check_log_file)
        self.label_list = label_list # 将传入的标签列表保存到类的实例变量label_list中

    # 定义__call__方法，使得Predict的实例可以像函数一样被调用
    # 该方法接收两个参数：x（键值对列表）和task_description（任务描述字符串）
    def __call__(self, x, task_description):
        # 通过列表推导式和字符串拼接构建提示字符串
        # 对于x中的每一个键值对(a, b)
        # 将其转换成格式"B: A "(其中 B 是键 b 的首字母大写形式）
        # 然后将所有这些片段合并成一个字符串
        prompt = "".join([f"{b.capitalize()}: {a} " for a, b in x])
        prompt = task_description + prompt + "Answer: " # 生成完整的字符串
        answer = self.query(prompt) # 发送完整的prompt到api接口得到结果
        # 对于每个标签，计算其在答案中的出现次数
        # 并从中减去带有前导下划线的同名标签的计数，这有助于消除潜在的误计数
        # 例如，如果我们要统计entailment出现的次数，那么肯定要除去not_entailment出现的次数
        counts = [
            answer.count(label) - answer.count(f"_{label}") for label in self.label_list
        ]

        # 计算counts列表中的最大值，以确定哪个标签的计数最多
        max_value = max(counts)
        # 找出所有计数为max_value的标签的索引
        max_indices = [i for i, value in enumerate(counts) if value == max_value]

        # 如果有一个明确的最大值索引（即列表中只有一个最大值），返回该索引
        # 如果有多个或无明确的最大值，返回None
        return max_indices[0] if len(max_indices) == 1 else None
