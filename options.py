import argparse

def get_args():
    # 创建了一个 ArgumentParser 对象 parser，它将用于处理命令行参数
    # description 参数用于提供这个命令行工具的简短描述，当在命令行中使用 -h 或 --help 选项时显示
    parser = argparse.ArgumentParser(description='For Project KGPA')

    parser.add_argument(
        '--log_file',
        type=str,
        help='用于LLMCalls/LLMLogSql.py中提供到sql数据库的连接',
    )

    parser.add_argument(
        '--API_key',
        type=str,
        help='用于给出调用Chatgpt模型需要的key,最早见于LLMCalls/LLMCall.py',
        default=''
    )

    parser.add_argument(
        '--API_base',
        type=str,
        help='用于给出Chatgpt服务器的网址,最早见于LLMCalls/LLMCall.py',
        default='https://api.openai.com/v1/chat', # 此行用于gpt-4-turbo, gpt-4o, gpt-4-0613
        # default = 'https://api.openai.com/v1', # 此行用于gpt-3.5-turbo-instruct
    )

    parser.add_argument(
        '--version',
        type=str,
        help='指定模型的版本,最早见于LLMCalls/LLMCall.py',
        default='gpt-4-turbo'
    )

    parser.add_argument(
        '--max_fail_count',
        type=int,
        help='Maximum number of fail attempts,最早见于LLMCalls/LLMCall.py',
        default=100,
    )

    parser.add_argument(
        '--pkl_file_path',
        type=str,
        help='以二进制读形式打开路径（如）info/{self.dataset}_info.pkl文件，最早见于Attack/Attack.py',
        default='info/mnli-m_info.pkl',
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="google_re",
        help="dataset [SST-2, QQP, MNLI-m, MNLI-mm, RTE, QNLI]，以及知识图谱数据集",
    )

    parser.add_argument(
        "--t2p_based_model",
        type=str,
        default="template",
        help="即选择使用模板还是llm将三元组转化为prompt，在template与llm之间选择"
    )

    parser.add_argument(
        "--triple_file_path",
        type=str,
        default="data/kg_examples/google_re",
        help="知识图谱数据集的主要文件位置"
    )

    parser.add_argument(
        "--tau_wmr",
        type=float,
        default=0.15,
        help="单词修改率word modification rate，范围：[0,1], 实际上没有用",
    )

    parser.add_argument(
        "--tau_bert",
        type=float,
        default=0.92,
        help="BERTScore，评分：[-1,1], 实际上没有用"
    )

    parser.add_argument(
        "--tau_llm",
        type=float,
        default=0.92,
        help="LLMScore，评分：[-1,1]"
    )

    parser.add_argument(
        "--pertub_type",
        type=int,
        default=7, # 7转述句子，想都不要想对kg比较有效
        help="指定使用的干扰类型的索引，可能表示不同的对抗性攻击策略范围：{0,1,2,3,4,5,6,7,8}",
    )

    parser.add_argument(
        "--t_a",
        type=int,
        default=0,
        help="指定句子被干扰的类型的索引",
    )

    # store_true是一个动作类型，用于处理布尔值的命令行参数
    # 当你使用store_true作为一个参数的动作时
    # 这意味着如果该命令行参数被指定了（也就是在命令行中出现了这个参数），那么相应的变量将被设置为True
    # 如果这个参数没有被指定，那么相应的变量则保持其默认值，通常为False
    parser.add_argument(
        "--few_shot",
        action="store_false",
        help="是否使用少样本策略"
    )

    parser.add_argument(
        "--ensemble",
        action="store_false",
        help="是否使用集成策略"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=" 指定每个数据批次的大小"
    )

    parser.add_argument(
        "--attack_log_file",
        type=str,
        default="attack.db",
        help="用于保存对抗性攻击结果的文件名",
    )

    parser.add_argument(
        "--check_log_file",
        type=str,
        default="check.db",
        help="file to save LLM check result",
    )

    parser.add_argument(
        "--kgb_fsa_log_file",
        type=str,
        default="kgb_fsa.db",
        help="file to save kgb_fsa result",
    )

    parser.add_argument(
        "--t2p_log_file",
        type=str,
        default="t2p.db",
        help="file to save t2p result",
    )

    parser.add_argument(
        "--pre_log_file",
        type=str,
        default="pre.db",
        help="file to save pre_log result",
    )

    # parse_known_args方法解析已知的参数，并返回两个值
    # 一个是命名空间，包含解析的参数，即args
    # 另一个是列表，包含那些不被识别的参数字符串。这里通过 _ 忽略了不被识别的参数列表。
    args, _ = parser.parse_known_args()

    return parser.parse_args()
