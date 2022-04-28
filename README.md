# Semi-supervised Sequence Learning

## 实验环境

- Pycharm 2021

* TensorFlow v2.8

## 端到端的IMDB情感分类

### 获取数据

```bash
$ wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz \
    -O /tmp/imdb.tar.gz
$ tar -xf /tmp/imdb.tar.gz -C /tmp
```

目录 `/tmp/aclImdb` 包含原始IMDB数据。

### 生成词汇

```bash
$ IMDB_DATA_DIR=/tmp/imdb
$ python gen_vocab.py \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=/tmp/aclImdb \
    --lowercase=False
```

词库以及频率相关的文件将在 `$IMDB_DATA_DIR` 中生成。

###  生成训练、验证以及测试数据

```bash
$ python gen_data.py \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=/tmp/aclImdb \
    --lowercase=False \
    --label_gain=False
```

`$IMDB_DATA_DIR` 包含TFRecords文件。

### 预训练IMDB语言模型

```bash
$ PRETRAIN_DIR=/tmp/models/imdb_pretrain
$ python pretrain.py \
    --train_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --num_candidate_samples=1024 \
    --batch_size=256 \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.9999 \
    --max_steps=100000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings
```

`$PRETRAIN_DIR` 包含预训练语言模型的检查点

### 训练分类器

大多数标志保持不变，除了删除候选采样和添加 `pretrained_model_dir`，分类器将从中加载预训练的嵌入和 LSTM 变量，以及与对抗性相关的标志训练和分类。

```bash
$ TRAIN_DIR=/tmp/models/imdb_classify
$ python train_classifier.py \
    --train_dir=$TRAIN_DIR \
    --pretrained_model_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --cl_num_layers=1 \
    --cl_hidden_size=30 \
    --batch_size=64 \
    --learning_rate=0.0005 \
    --learning_rate_decay_factor=0.9998 \
    --max_steps=15000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings \
    --adv_training_method=vat \
    --perturb_norm_length=5.0
```

### 评估测试数据

```bash
$ EVAL_DIR=/tmp/models/imdb_eval
$ python evaluate.py \
    --eval_dir=$EVAL_DIR \
    --checkpoint_dir=$TRAIN_DIR \
    --eval_data=test \
    --run_once \
    --num_examples=25000 \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --batch_size=256 \
    --num_timesteps=400 \
    --normalize_embeddings
```

## 相关总结

主要入口点是下面列出的二进制文件。每个二进制文件训练都会构建一个 ` VatxtModel` （ `VatxtModel` 都定义在 `graphs.py` 中）。`VatxtModel` 使用图形构建块在 `inputs.py` 中定义（定义输入数据读取和解析） `layers.py` （定义核心模型组件）和 `adversarial_losses.py` （定义对抗性训练损失）。训练循环本身定义在 `train_utils.py` 中。

### 二进制文件

*   预训练： `pretrain.py`
*   分类器训练： `train_classifier.py`
*   评估： `evaluate.py`

### 命令行标志

与分布式训练和训练循环本身相关的标志定义在 `train_utils.py` 文件中。

与模型超参数相关的标志定义在 `graphs.py` 文件中。

与对抗性训练相关的标志定义在 `adversarial_losses.py` 文件中。

每个作业特有的标志在主二进制文件中定义。

### 数据生成

*   词汇生成： `gen_vocab.py`
*   数据生成： `gen_data.py`

`document_generators.py` 中定义了命令行标志控制处理哪个数据集以及如何如何处理。
