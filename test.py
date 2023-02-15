# author:Wang Yanguang
# time:2023/2/15 15:06
#定义推断函数
from torch.utils.data import DataLoader
from transformers import BertTokenizer, logger
import NER_config
import torch

from BERTner import NERDataset, BertNER


def infer_function(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(NER_config.robert_model, do_lower_case=True, skip_special_tokens=True)
    id2label = NER_config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0
    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            #loss = model((batch_data, batch_token_starts),
                         #token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            #dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            #batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
    return pred_tags

def new_infer(text):
    words = list(text)
    label = ['O'] * len(words)
    word_list = []
    label_list = []
    word_list.append(words)
    label_list.append(label)
    output_filename = ''
    np.savez_compressed(output_filename,words = word_list, lables = label_list)
    #重新加载
    data = np.load(output_filename, allow_pickle=True)
    word_test = data["words"]
    label_test = data["lables"]
    test_dataset = NERDataset(word_test, label_test, NER_config)
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=NER_config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    # Prepare model
    if NER_config.model_dir is not None:
        #model = torch.load(NER_config.model_dir)
        model = BertNER.from_pretrained(NER_config.model_dir)
        model.to(NER_config.device)
        logger.info("--------Load model from {}--------".format(NER_config.model_dir))
    else:
        logger.info("--------No model to test !--------")
        return
    pre_tegs = infer_function(test_loader, model, mode='test')
    return pre_tegs


text = '2022年11月，马来西亚随荷兰国家队征战2022年卡塔尔世界杯'
pre_tegs = new_infer(text)

# 取出位置
start_index_list = []
end_index_list = []
for index in range(len(pre_tegs[0])):
    if index != 0 and pre_tegs[0][index] != 'O' and pre_tegs[0][index - 1] == 'O':
        start_index = index
        start_index_list.append(start_index)
    if index != len(pre_tegs[0]) - 1 and pre_tegs[0][index] != 'O' and pre_tegs[0][index + 1] == 'O':
        end_index = index
        end_index_list.append(end_index)
    if index == 0 and pre_tegs[0][index] != 'O':
        start_index = index
        start_index_list.append(start_index)
    if index == len(pre_tegs[0]) - 1 and pre_tegs[0][index] != 'O':
        end_index = index
        end_index_list.append(end_index)
# 展示
for index in range(len(start_index_list)):
    print(text[start_index_list[index]:end_index_list[index] + 1])