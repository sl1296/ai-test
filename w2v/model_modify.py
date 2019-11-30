import torch
import csv
import math
from tqdm import tqdm
import os
import time
import torch.optim as optim
import pandas as pd
import torch.utils.data as Data
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
from transformers import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn


class BertForMultipleChoice(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # fix 住 Linear 层以外的
        for p in self.parameters():
            p.requires_grad = False

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # https://github.com/huggingface/transformers/blob/master/examples/contrib/run_swag.py

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_features_from_task_2(questions_path, answers_path, tokenizer, max_seq_length):
    questions = pd.read_csv(questions_path)
    answers = pd.read_csv(answers_path, header=None, names=['id', 'ans'])
    data = pd.merge(questions, answers, how='left', on='id')

    features = []
    for i in data.values:
        context_tokens = tokenizer.tokenize(i[1])
        choices_features = []
        for j in range(2, 5):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            # 如数据中的 5618，没有第三个选项，因此读出来可能是 nan，这里 str 强制转换处理一下
            ending_tokens = tokenizer.tokenize(str(i[j]))

            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))
        features.append((choices_features, ord(i[5]) - 65))
    return features


def train(model, train_data, optimizer):
    model.train()
    pbar = tqdm(train_data)
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x[:, :, 0], attention_mask=x[:, :, 1], token_type_ids=x[:, :, 2], labels=y)
        loss = output[0]
        loss.backward()
        optimizer.step()

        # 得到预测结果
        pred = output[1].argmax(dim=1, keepdim=True)
        # 计算正确个数
        correct = pred.eq(y.view_as(pred)).sum().item()
        pbar.set_postfix({'loss': '{:.3f}'.format(loss.item()), 'acc': '{:.3f}'.format(correct * 1.0 / len(x))})


def test(model, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for step, (x, y) in enumerate(test_data):
            x, y = x.to(device), y.to(device)
            output = model(x[:, :, 0], attention_mask=x[:, :, 1], token_type_ids=x[:, :, 2], labels=y)
            test_loss += output[0].item()
            pred = output[1].argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def create_datasets(features, shuffle=True):
    """
    使用 features 构建 dataset
    :param features:
    :param shuffle: 是否随机顺序，默认 True
    :return:
    """
    x = []
    y = []
    for i in features:
        res = []
        # 存储每个问题选择题的 input_ids, input_mask, segment_ids
        for j in range(3):
            res.append(i[0][j][1:])
        x.append(res)
        y.append(i[1])
    x = torch.tensor(x)
    y = torch.tensor(y)
    if shuffle:
        perm = torch.randperm(len(features))
        x = x[perm]
        y = y[perm]
    return Data.TensorDataset(x, y)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    args = {
        'batch_size': 20,
        'test_batch_size': 20,
        'lr': 0.001,
        'fine_tune_lr': 0.00001,
        'epochs': 10,
        'fine_tune_epochs': 10,
        'log_interval': 10,
        'use_cuda': torch.cuda.is_available(),
        'data_loader_num_workers': 4,
        'split_rate': 0.7,
        'max_seq_length': 20,
    }
    # logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    # Load pre-trained model (weights)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    config = BertConfig.from_json_file('pre_weights/bert-base-cased_config.json')
    model = BertForMultipleChoice.from_pretrained('pre_weights/bert-base-cased_model.bin', config=config)

    # tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    # config = BertConfig.from_json_file('pre_weights/bert-large-cased_config.json')
    # model = BertForMultipleChoice.from_pretrained('pre_weights/bert-large-cased_model.bin', config=config)

    # Set the model in evaluation mode to desactivate the DropOut modules
    # This is IMPORTANT to have reproductible results during evaluation!
    device = torch.device('cuda:0' if args['use_cuda'] else 'cpu')
    kwargs = {'num_workers': args['data_loader_num_workers'], 'pin_memory': True} if args['use_cuda'] else {}

    model.to(device)

    features = get_features_from_task_2(
        'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training  Data/subtaskB_data_all.csv',
        'SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/Training  Data/subtaskB_answers_all.csv',
        tokenizer, 20)
    dataset = create_datasets(features, shuffle=True)
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(dataset[:int(len(dataset) * args['split_rate'])][0],
                                   dataset[:int(len(dataset) * args['split_rate'])][1]),
        batch_size=args['batch_size'],
        shuffle=True,
        **kwargs
    )
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(dataset[int(len(dataset) * args['split_rate']):][0],
                                   dataset[int(len(dataset) * args['split_rate']):][1]),
        batch_size=args['test_batch_size'],
        shuffle=True,
        **kwargs
    )
    train_data = list(train_loader)
    test_data = list(test_loader)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['lr'])

    start_time = time.time()
    # 先对预训练模型后几层进行训练
    print('start train...')
    for epoch in range(args['epochs']):
        print('Epoch {}/{}'.format(epoch + 1, args['epochs']))
        train(model, train_data, optimizer)
        test(model, test_data)
    print('Total Time: ', time.time() - start_time)

    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=args['fine_tune_lr'])

    start_time = time.time()
    # 整体进行 fine-tune
    print('start fine-tune...')
    for epoch in range(args['fine_tune_epochs']):
        print('Epoch {}/{}'.format(epoch + 1, args['fine_tune_epochs']))
        train(model, train_data, optimizer)
        test(model, test_data)
    print('Total Time: ', time.time() - start_time)
