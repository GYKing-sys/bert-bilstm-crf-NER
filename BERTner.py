#导入必要的库
import os
import json
from random import random
import numpy as np
from transformers.models.bert.modeling_bert import *
from tqdm import tqdm
import torch.nn as nn
from loguru import logger
from sklearn.metrics import accuracy_score,recall_score,f1_score
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from sklearn.model_selection import train_test_split
import warnings

import NER_config

warnings.filterwarnings('ignore')
#生成日志
log_path = ""
logger.add(log_path + 'Train.log', format="{time} {level} {message}", level="INFO")


# 数据处理
def Data_preprocess(input_filename, output_filename):
    count = 0
    word_list = []
    label_list = []
    with open(input_filename, 'r') as reader:
        lines = reader.readlines()
    random_list = []
    # 选取12000条数据
    for _ in tqdm(range(12000)):  # 12000
        # 设定随机值,进行随机选取
        random_index = random.randint(1, 4495464)  # 测试499497 #训练集4495465
        if random_index not in random_list:
            random_list.append(random_index)
            json_line = json.loads(lines[random_index].strip())
            text = json_line['text']
            # 设定了选取长度
            if len(text) <= 510:
                words = list(text)
                label_entity = json_line.get('label', None)
                # label先全部设为"O"
                label = ['O'] * len(words)
                # 判断如果不等于None
                if label_entity is not None:
                    count += 1
                    for key, value in label_entity.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                # 判断是否超出边界，做一个判断
                                if ''.join(words[start_index:end_index + 1]) == sub_name:
                                    # 单实体标注S-entity
                                    if start_index == end_index:
                                        label[start_index] = 'S-' + key
                                    else:
                                        # 多字实体采用B-entity I-entity E-entity 的标注方式
                                        label[start_index] = "B-" + key
                                        label[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                                        label[end_index] = 'E-' + key

                word_list.append(words)
                label_list.append(label)
            else:
                continue
    print(len(word_list), len(label_list))
    # 保存成二进制文件
    np.savez_compressed(output_filename, words=word_list, lables=label_list)
    # 统计处理数量
    print(count)

# train_input = ''
# train_output = ''
# word_list,label_list = Data_preprocess(train_input,train_output)
#
# test_input = ''
# test_output = ''
# Data_preprocess(test_input,test_output)


class NERDataset(Dataset):
    def __init__(self, words, labels, config, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer.from_pretrained(config.robert_model, do_lower_case=True)
        self.label2id = config.label2id
        self.id2label = {_id: _label for _label, _id in list(config.label2id.items())}
        self.dataset = self.preprocess(words, labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config.device

    def preprocess(self, origin_sentences, origin_labels):
        data = []
        sentences = []
        labels = []
        for line in tqdm(origin_sentences):
            # replace each token by its index
            # we can not use encode_plus because our sentences are aligned to labels in list type
            words = []
            word_lens = []
            for token in line:
                # bert对字进行编码转化为id表示
                words.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            # 变成单个字的列表，开头加上[CLS]
            words = ['[CLS]'] + [item for token in words for item in token]
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))
        for tag in origin_labels:
            label_id = [self.label2id.get(t) for t in tag]
            labels.append(label_id)
        for sentence, label in zip(sentences, labels):
            if len(sentence[0]) - len(label) == 1:
                data.append((sentence, label))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        # batch length
        batch_len = len(sentences)
        # compute length of longest sentence in batch
        max_len = max([len(s[0]) for s in sentences])
        max_label_len = 0  # 改动前max_label_len = 0
        # padding data 初始化
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts = []
        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            batch_labels[j][:cur_tags_len] = labels[j]
        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_data, batch_label_starts, batch_labels]


class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        # 定义分类类别，也可以写在加载预训练模型的config文件中
        self.num_labels = 5
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=config.lstm_embedding_size,  # 1024
            hidden_size=config.hidden_size // 2,  # 1024
            batch_first=True,
            num_layers=2,
            dropout=config.lstm_dropout_prob,  # 0.5
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        # 将结果送入bilstm，再次提取特性
        lstm_output, _ = self.bilstm(padded_sequence_output)
        # 将lstm的结果送入线性层，进行五分类
        logits = self.classifier(lstm_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            # 将每个标签的概率送入到crf中进行解码，并获得loss
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs
        # contain: (loss), scores
        return outputs

#定义训练函数
def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    # 设定训练模式
    model.train()
    train_losses = 0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels = batch_samples
        batch_masks = batch_data.gt(0)  # get padding mask
        # 计算损失值
        loss = model((batch_data, batch_token_starts),
                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
        train_losses += loss.item()
        #梯度更新
        model.zero_grad()
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=NER_config.clip_grad)
        # 计算梯度
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_loader)
    logger.info("Epoch: {}, train loss: {}",epoch, train_loss)

#根据预测值和真实值计算评价指标
def compute_acc_recall(batch_output,batch_tags):
    acc = 0
    recall = 0
    f1 = 0
    for index in range(len(batch_output)):
        acc += accuracy_score(batch_output[index],batch_tags[index])
        recall += recall_score(batch_output[index],batch_tags[index],average='macro')
        f1 += f1_score(batch_output[index],batch_tags[index],average='macro')
    return (acc/len(batch_output),recall/len(batch_output),f1/len(batch_output))

#定义验证函数
def evaluate(dev_loader, model, mode='dev'):
    # 设置为模型为验证模式
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(NER_config.robert_model, do_lower_case=True, skip_special_tokens=True)
    id2label = NER_config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0
    with torch.no_grad():
        for idx, batch_samples in tqdm(enumerate(dev_loader)):
            batch_data, batch_token_starts, batch_tags = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[idx for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[idx for idx in indices if idx > -1] for indices in batch_tags])
            #pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            #true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])
    assert len(pred_tags) == len(true_tags)
    # logging loss, f1 and report
    metrics = {}
    acc , recall, F1= compute_acc_recall(true_tags,pred_tags)
    metrics['acc'] = acc
    metrics['recall'] = recall
    metrics['f1'] = F1
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics

def test(NER_config):
    data = np.load(NER_config.test_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    test_dataset = NERDataset(word_test, label_test, NER_config)
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=NER_config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    # Prepare model
    if NER_config.model_dir is not None:
        model = BertNER.from_pretrained(NER_config.model_dir)
        model.to(NER_config.device)
    val_metrics = evaluate(test_loader, model, mode='test')
    logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_metrics['F1']))

def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, NER_config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch)
        #开始验证
        val_metrics = evaluate(dev_loader, model, mode='dev')
        val_f1 = val_metrics['f1']
        logger.info("Epoch: {}, dev loss: {}, f1 score: {}",epoch, val_metrics['loss'], val_f1)
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            #模型保存需要更改
            torch.save(model,model_dir)
            logger.info("--------Save best model!--------")
            if improve_f1 < NER_config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= NER_config.patience_num and epoch > NER_config.min_epoch_num) or epoch == NER_config.epoch_num:
            logger.info("Best val f1: {}",best_val_f1)
            break
    logger.info("Training Finished!")

def dev_split(dataset_dir):
    """从训练集合中划分验证集和训练集"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["lables"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=0.01, random_state=0)
    return x_train, x_dev, y_train, y_dev


def run(config):
    """train the model"""
    # 处理数据，
    # 分离训练集、验证集
    word_train, word_dev, label_train, label_dev = dev_split('train')
    # 创建dataset
    train_dataset = NERDataset(word_train, label_train, config)
    dev_dataset = NERDataset(word_dev, label_dev, config)
    # get dataset size
    train_size = len(train_dataset)
    # 创建dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)

    # 实例化模型
    device = config.device
    model = BertNER.from_pretrained(config.roberta_model, num_labels=len(config.label2id))
    model.to(device)
    # Prepare optimizer
    if config.full_fine_tuning:
        # model.named_parameters(): [bert, bilstm, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        lstm_optimizer = list(model.bilstm.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)



if __name__ == '__main__':
    run(NER_config)
    test(NER_config)

