# 计算模型的预测准确率

def get_accuracy(pred, label):
    assert len(pred) == len(label), "Prediction and label should have same length"
    correct = [i == j for (i, j) in zip(pred, label)]
    return sum(correct) / len(correct)
