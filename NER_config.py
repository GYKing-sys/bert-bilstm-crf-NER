# author:Wang Yanguang
# time:2023/2/11 22:36

import torch

robert_model = ''
model_dir = ''
# 训练集、验证集划分比例
dev_split_size = 0.1
# 是否加载训练好的NER模型
load_before = False
# 指定device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 10
epoch_num = 150
min_epoch_num = 5
patience = 0.0002
patience_num = 10

labels = ['entity']
label2id = {
    "O": 0,
    "B-entity": 1,
    "I-entity": 2,
    "E-entity": 3,
    "S-entity": 4,
}

id2label = {_id: _label for _label, _id in list(label2id.items())}

# BertNER的超参数
# num_labels = len(label2id)
# hidden_dropout_prob = 0.3
# lstm_embedding_size = 768
# hidden_size = 1024
# lstm_dropout_prob = 0.5