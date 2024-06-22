# 计算攻击成功率（Attack Success Rate）

# pred：原始预测结果列表
# adv_pred：经过对抗攻击后的预测结果列表
# label：真实标签列表
def get_ASR(pred, adv_pred, label):
    assert len(pred) == len(label) and len(pred) == len(adv_pred), \
        "len(pred) must equal len(label), len(adv_pred) must equal len(label)"
    # 下面代码的逻辑是
    # ASR = 在对抗攻击后预测变为错误的项的数量/原始预测是正确的项的数量
    correct = [i == j for (i, j) in zip(pred, label)]
    adv_wrong = [
        (pred[i] == label[i] and adv_pred[i] != label[i]) for i in range(len(label))
    ]
    return sum(adv_wrong) / sum(correct)
