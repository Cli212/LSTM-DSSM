

### 问题描述

FAQ：（Frequently asked Questions），是检索式问答系统。通常情况是给定标准问题库，系统需要将用户输入的query匹配用户最想问的问题上。用户输入的query通常是短文本，标准问题库是一个封闭的集合。每个标准问题都有固定答案和标题，同时会有多个扩展问法和关键词。模型所需要解决的是给定query，找到标准问题里用户最接受的答案。

解决FAQ问题通常有两种思路：1）相似问题匹配，即计算用户问题与现有知识库中的问题的相似度，返回用户问题对应的最精准的答案；2）问题答案匹配，即计算用户问题与知识库中答案的匹配度，返回用户问题对应的最精准的答案，该思路是选择答案，及QA匹配。

### 算法介绍

#### DSSM类

DSSM （Deep Structured Semantic Models）的原理很简单，通过搜索引擎里 Query 和 Title的海量的点击曝光日志（FAQ场景下对应着标准问和用户问），用 DNN 把 Query 和 Title 表达为低纬语义向量，并通过 cosine 距离来计算两个语义向量的距离，最终训练出语义相似度模型。该模型既可以用来预测两个句子的语义相似度，又可以获得某句子的低纬语义向量表达。

DSSM 从下往上可以分为三层结构：输入层、表示层、匹配层

1. 输入层

   中文的输入层处理方式有两种：单字输入和分词输入，通过seg参数控制，seg=True为分词输入，seg=False为单字输入。

2. 表示层

   DSSM 的表示层采用 BOW（Bag of words）的方式，相当于把字向量的位置信息抛弃了，整个句子里的词都放在一个袋子里了，不分先后顺序。

   紧接着是一个含有多个隐层的 DNN，如下图所示：

   ![1596684616334](images/1596684616334.png)

   用$W_i$表示第 i 层的权值矩阵，$b_i$ 表示第 i 层的 bias 项。则第一隐层向量 l1（300 维），第 i 个隐层向量 li（300 维），输出向量 y（128 维）可以分别表示为：

   用 tanh 作为隐层和输出层的激活函数：

   最终输出一个 128 维的低纬语义向量。

3. 匹配层

   Query 和 Doc 的语义相似性可以用这两个语义向量(128 维) 的 cosine 距离来表示：

   通过softmax 函数可以把Query 与正样本 Doc 的语义相似性转化为一个后验概率：

   其中 r 为 softmax 的平滑因子，D 为 Query 下的正样本，D-为 Query 下的负样本（采取随机负采样），D 为 Query 下的整个样本空间。

   在训练阶段，通过极大似然估计，我们最小化损失函数：

   $L=-log\prod\limits_{(Q,D^+)}{P(D^+|Q)}$

   残差会在表示层的 DNN 中反向传播，最终通过随机梯度下降（SGD）使模型收敛，得到各网络层的参数{ $W_i$,$b_i$ }。

LSTM-DSSM针对DSSM表示层无法捕获上下文特征的缺点，使用加入了peephole的LSTM代替DNN对句子进行向量表示。加入peephole的LSTM其实就是将上一个时间步的单元状态$c_{t-1}$也作为时刻t的输入，结构如下： 

![1596685072182](images/1596685072182.png)

#### Siamese类

简单来说，Siamese network就是“孪生神经网络”，神经网络的“孪生”是通过共享权值来实现的。

通过一个共享权值的表示层，比如LSTM或者DNN对语句对进行向量表示，分别得到语句对中两个句子的向量表示，通过比较两个向量的距离E（曼哈顿距离、欧氏距离等），得到语义相似度。模型结构图如下：



![1596689313588](images/1596689313588.png)

损失函数由两部分组成，对于正例的损失函数和对于负例的损失函数：<br>
![](https://latex.codecogs.com/svg.latex?L_+(x_1,x_2)%20=%20\frac{1}{4}(1-E_w)^2)<br>
![](https://latex.codecogs.com/svg.latex?L_-(x_1,x_2)%20=%20\begin{cases}%20E_w^2%20\quad%20if%20E_w%3Em\\0%20\quad%20otherwise\end{cases}) 
![](https://latex.codecogs.com/svg.latex?L%20=%20yL_+(x_1,x_2)+(1-y)L_-(x_1,x_2))<br>

$L_+(x_1,x_2) = \frac{1}{4}(1-E_w)^2$

$L_-(x_1,x_2) = \begin{cases} E_w^2 \quad if E_w>m\\0 \quad otherwise\end{cases} $

$L = yL_+(x_1,x_2)+(1-y)L_-(x_1,x_2)$ 

### 使用方法	

共包含4个py文件，data_preprocess.py、DSSM.py、train.py、predict.py，data_preprocess.py用来预处理数据以及建词典。传入数据为csv或excel文件，一列标准问一列用户问，多个用户问之间以、分割，数据格式如下：

![1596764906219](images/1596764906219.png)

__data_preprocess.py 参数__

~~~
train_data_path   训练数据文件路径
test_data_path     测试数据文件路径
val_data_path       验证数据文件路径
vocab_path            建好的字典保存路径
seg                              建字典时是否分词
~~~

例：

~~~~~~powershell
python data_preprocess.py --train_data_path data/faq_train_data.xlsx --test_data_path data/faq_test_data.xlsx --vocab_path data/vocab.txt --seg True
~~~~~~

__train.py 参数__

~~~
train_data_path   训练数据文件路径，data_preprocess产生的txt文件
test_data_path     测试数据文件路径，data_preprocess产生的txt文件
vocab_path            字典文件路径，data_preprocess产生的txt文件
seg                              处理数据是否分词，seg要与data_preprocess建字典时一致
emb_dim                 词向量维度，默认128
model_dim             模型表示层的维度，DNN或者LSTM的隐层维度，默认256
dropout                     0~1，默认0.0
neg                              负采样大小，默认30
max_len                    句子的最大长度，句子长度大于max_len会被截断，小于max_len会进行padding，默认15
epoch                         训练轮数，默认10
batch                           mini-batch 大小，默认16
layer_class                表示层的种类，lstm或dnn，默认dnn
learning_rate           学习率，默认3e-4
~~~

例：

~~~powershell
python train.py --train_data_path data/train_data.txt --vocab_path data/vocab.txt --neg 90 --test_data_path data/test_data.txt --epoch 20 --dropout 0.1 --emb_dim 256 --model_dim 400
~~~

训练完之后，会在model文件夹下生成名为model_emb_dim_model_dim_neg_maxlen的模型文件

__predict.py__

加载Predict类，读取模型文件并且进行预测

~~~python
p = Predict(model_path='model/model_300_512_30_15',vocab_file='data/vocab.txt',seg=True)
p.predict('CA证书的用处')
~~~

### Experiment

对于DSSM类模型来说，负采样的数量非常重要，负采样数量增加对结果的提升非常明显，原因是本份数据集中不同标准问之间语义可能会有近似的情况，所以如果只去提升正向样本与标准问的相似度很容易就会在测试时产生匹配偏差。本份数据集有137类，实验中效果较好的负采样数量为70-80,。

dropout尽量设置为0.0或者是0.1，因为数据量小，dropout效果不好，还会增加训练时间，我们只需要使用正则化的方式防止过拟合即可。

训练轮数epoch选择范围20-30达到的效果最好，轮数太多可能发生过拟合。

关于模型表示层选择DNN还是LSTM的这个问题，本次实验DNN的效果更好，且能在很短的时间内完成训练。虽然LSTM在本份数据集上表现不如DNN，主要原因还是训练LSTM时间太久，导致负采样值不能调太高，模型参数量也不能调太大。而且训练数据也少，且都是小短句，无法发挥出LSTM的优势，如果之后有大数据量的数据集还是可以用LSTM实验一下的。

在这份FAQ数据集上，分词或不分词效果差别不大，甚至在其他超参数固定的情况下，不分词训练的效果要好于分词训练的效果。

learning rate选择范围大于1e-3小于1e-5即可，影响不大。


### To Do List
1. 如果数据中包含问题答案，可以尝试计算用户问题与知识库中答案的匹配度。
2. 使用其他种类的距离度量方法来衡量向量之间的距离，比如曼哈顿距离、欧式距离等。论文原文中使用的是cosine距离，但cosine距离容易导致梯度消失的问题可能会对训练造成影响。
3. Siamese CBOW是借鉴CBOW模型的想法提出的Siamese类模型，与以上模型不同的一点是Siamese CBOW是无监督的算法，如果训练数据量大可以实现Siamese CBOW试一下效果。
4. DSSM表示层使用Transformer结构。


### 参考论文

[Huang, Po-Sen, He, Xiaodong, Gao, Jianfeng, Deng, Li, Acero, Alex, and Heck, Larry. Learning
deep structured semantic models for web search using clickthrough data. In Proceedings of the
22Nd ACM International Conference on Conference on Information &#38; Knowledge Manage-
ment, CIKM ’13, pp. 2333–2338. ACM, 2013.](https://dl.acm.org/doi/10.1145/2505515.2505665)

[Palangi, H., et al. (2014) *Semantic* Modelling *with* *Long-Short-Term* *Memory* *for* Information *Retrieval*. arXiv1412.6629](https://arxiv.org/pdf/1412.6629.pdf)

[Neculoiu, P., Versteegh, M. and Rotaru, M. (2016) Learning Text Similarity with Siamese Recurrent Networks. Proceedings of the 1st Workshop on Representation Learning for NLP, Berlin, 11 August 2016, 148-157.](https://www.aclweb.org/anthology/W16-1617.pdf)
