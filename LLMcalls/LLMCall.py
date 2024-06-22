import time
import openai
import logging
from LLMcalls.LLMLogSql import LLMLogSql
from ratelimit import limits, sleep_and_retry

"""
# 引入令牌桶算法，限制询问速率
from token_bucket import Limiter
from token_bucket.storage import MemoryStorage

# 创建内存存储实例
storage = MemoryStorage()

# 使用内存存储创建令牌桶限流器
limiter = Limiter(8, 100, storage=storage)
"""

"""
关于response，例子: ChatGPT对问题"What is AI?"给出回答，得到的response格式如下
{
  "choices": [
    {
      "message": {
        "content": "AI, or Artificial Intelligence, \
        refers to the simulation of human intelligence in machines \
        that are programmed to think like humans and mimic their actions."
      }
    }
  ]
}
"""

# 用于调用大模型
class LLMCall(LLMLogSql):
    # 下面两个值是类变量而不是实例变量（self.log_count与self.save_count）
    # 下面两个值不设置为实例变量是为了在所有LLMCall的实例中共享相同的值
    log_count: int = 0 # 追踪从数据库中成功读取并返回的响应次数
    save_count: int = 0 # 追踪新从API获取并存入数据库的响应次数

    def __init__(
            self, args, log_file
    ) -> None: # REVISE
        # args是一个句柄，这里主要用来传递如下值
        # 调用Chatgpt模型需要的API_key
        # 给出Chatgpt服务器的网址API_base
        # 指定模型的版本version
        # fail_count可以达到的最大值max_fail_count（默认为100）
        super().__init__(log_file)
        self.API_key = args.API_key
        self.API_base = args.API_base
        self.version = args.version
        self.fail_count = 0 # 为了避免死循环，当call与query互相调用加起来达到一定次数后终止进程
        self.max_fail_count = args.max_fail_count

    @sleep_and_retry
    @limits(calls=200, period=60)  # 设定每分钟最多200次API调用
    def call(self, prompt):
        # REVISE
        # 检测fail_count是否达到最大值
        # if self.fail_count >= self.max_fail_count:
        #     exit("REACH MAX FAIL COUNT")

        # openai库api的url，确保api请求能发送到正确服务器
        openai.api_base = self.API_base
        openai.api_key = self.API_key
        try:
            if self.version == 'gpt-3.5-turbo-instruct':
                response = openai.Completion.create(
                    model=self.version, # 指定了使用的模型版本，这个值在类初始化时设置
                    temperature=0, # 设置了生成响应的确定性级别，温度为0意味着生成最确定性的响应
                    # messages=[{"role": "user", "content": prompt}], # 此行用于gpt-4-turbo, gpt-4o, gpt-4-0613
                    prompt=prompt, # 此行用于gpt-3.5-turbo-instruct
                )
            else:
                response = openai.Completion.create(
                    model=self.version,  # 指定了使用的模型版本，这个值在类初始化时设置
                    temperature=0,  # 设置了生成响应的确定性级别，温度为0意味着生成最确定性的响应
                    messages=[{"role": "user", "content": prompt}],  # 此行用于gpt-4-turbo, gpt-4o, gpt-4-0613
                    # prompt=prompt, # 此行用于gpt-3.5-turbo-instruct
                )
        except Exception as e:
            # 使用logging库记录警告信息，将捕获的异常e记录到日志中
            logging.warning(e)
            # 如果发生异常，程序会暂停执行2秒
            # 为了防止在出现网络问题时过快重试而导致的连续失败
            time.sleep(2)
            # REVISE
            self.fail_count += 1
            return self.query(prompt)

        if self.version == 'gpt-3.5-turbo-instruct':
            return response["choices"][0]["text"] # 此行用于gpt-3.5-turbo-instruct
        else:
            return response["choices"][0]["message"]["content"]  # 此行用于gpt-4-turbo, gpt-4o, gpt-4-0613

    def query(self, prompt):
        # REVISE
        # 检测fail_count是否达到最大值
        # if self.fail_count >= self.max_fail_count:
        #     exit("REACH MAX FAIL COUNT")
        """
        # 通过漏桶算法限制询问速率
        while not limiter.consume("api_request", 1):  # 消耗一个令牌
            time.sleep(1)  # 如果没有可用令牌，则等待
        """

        # 首先尝试通过DBQuery方法查询本地数据库是否已存储对应于输入prompt的响应

        # 对于variable := expression
        # expression的结果会被赋值给variable
        # 同时expression的结果也会作为整个表达式的结果返回
        # 使得variable可以在更大的表达式或条件中立即使用

        # 如果存在，直接返回该响应并增加日志计数器log_count
        # if save_response := self.DBQuery(prompt):
        #     self.log_count = self.log_count + 1
        #     return save_response
        # 如果不存在，调用call方法从API获取响应
        response = self.call(prompt) # 如果不设置终止条件，call与query可能会互相反复调用，死循环
        self.DBInsert(prompt, response)
        self.save_count = self.save_count + 1
        # print(self.log_count / (self.save_count+self.log_count))
        response = response.replace("\n", "") # 去除所有换行符
        response = response.lstrip(' ')  # 去除开头的空格
        return response
