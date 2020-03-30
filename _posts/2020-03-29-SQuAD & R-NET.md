---
layout:     post
title:      SQuAD & R-NET 机器阅读理解
subtitle:   R-NET
date:       2020-03-28
author:     BY
header-img: img/post-bg-desk.jpg
catalog: true
mathjax: true
tags:
    - Python
    - TensorFlow
    - BERT
    - NLP
    - SQuAD
    - R-NET
    - 机器阅读理解
---




## 背景介绍:


### [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/)

![-w1783](http://roger-markdown.oss-cn-beijing.aliyuncs.com/2020/03/28/15852187516501.jpg)

### R-Net
![-w964](http://roger-markdown.oss-cn-beijing.aliyuncs.com/2020/03/28/15852188525905.jpg)
### 效果
![-w1394](http://roger-markdown.oss-cn-beijing.aliyuncs.com/2020/03/28/15852192061150.jpg)


## 1. SQuAD
斯坦福大学自然语言计算组发布SQuAD数据集，诸多团队参与其中，而微软亚研的R-NET是首个在某些指标中接近人类的深度学习模型。开源的CNTK版的[R-NET](jyh764790374/R-Net-in-CNTK)，
![-w1235](http://roger-markdown.oss-cn-beijing.aliyuncs.com/2020/03/28/15853629053994.jpg)


- 先介绍一下SQuAD数据集的特点，SQuAD数据集包含10w个样例，每个样例大致由一个三元组构成（文章Passage, 相应问题Question, 对应答案Answer), 以下皆用（P,Q,A）表示。

本篇论文介绍了R-NET,一个end-to-end神经网络模型用来解决阅读理解的问答，也就是旨在根据指定文章回答问题。首先，用gated attention-based recurrent networks来获得question-aware段落表示。然后，提出一种self-matching attention 机制来完善表示，就是将匹配段落与自身相匹配，可以有效地编码整个段落的信息。最后，利用pointer networks来定位段落中答案的位置。模型在SQuAD和MS-MARCO数据集上进行了大量的实验，并取得了最优效果。


## 介绍
本篇文章解决的是阅读理解中的问答，采用的数据集是SQuAD和MS-MARCO。其中SQuAD是根据问题在段落中找到answer span，而MS-MARCO提供了从Bing Index收集的相关文档，问题的答案可能在段落中也可能不在段落中。

![-w1084](http://roger-markdown.oss-cn-beijing.aliyuncs.com/2020/03/28/15852172731736.jpg)


模型主要包括4部分：
1. 利用recurrent network来分别编码问题、段落的表示；
2. gated matching layer来匹配问题和段落；
3. self-matching layer来整合整个段落的信息；
4. 基于pointer-network的答案边界预测layer。

SQuAD上大多数的问题都可以通过很简单的pattern匹配回答出来。深度学习确实是对于简单的组合型pattern学习有着非常强大的优势

## 贡献
本文工作的贡献主要有3个方面。

第一，提出了一种gated attention-based recurrent network，也就是在经典的attention-based recurrent networks上额外增加了门机制，这样做的主要原因是针对阅读理解的问题，段落中的每个单词的重要性是不同的。段落中的每个词语与其对应的question文本的attention权重被一起编码来产生question-aware段落表示。通过介绍门机制，我们的模型根据段落与问题的相关程度，赋予了段落不同的重要程度，掩盖了段落中不相关的部分。

第二，介绍了self-matching机制，这种机制可以有效地整合整个段落的信息来推断答案。通过gated matching layer，question-aware段落表示有效地编码了段落中每个单词的问题信息。然而由于RNN本身只能存储有限段落信息，一个候选答案通常不知道段落的其余部分的信息。为了解决这个问题，我们提出self-matching layer，用整个段落的信息动态完善段落表示。基于question-aware段落表示，在段落上采用gated attention-based recurrent networks来against段落，从这个段落来获取与当前段落词语相关的信息。Gated attention-based recurrent network layer和self-matching layer动态地丰富了每个段落表示，其中包含来自问题和段落的信息，使后续网络能够更好地预测答案。

本论文提出的方法取得了显著的效果，单个模型在SQuAD数据集实现了72.3%，ensemble模型则实现了76.9%。模型在MS-MARCO数据集取得了最优效果。


## R-NET模型结构

![-w1084](http://roger-markdown.oss-cn-beijing.aliyuncs.com/2020/03/28/15852172731736.jpg)


上图是R-NET模型的整体结构。首先，问题和段落分别经过双向RNN的处理。然后，用gated attention-based recurrent networks来匹配问题和段落，来获得question-aware段落表示。之后，采用self-matching attention来整合整个段落的信息、完善段落表示。最后，输入output layer来预测答案的边界。

### question and passage encoder
这里主要是将问题 $Q=\left\{w_{t}^{Q}\right\}_{t=1}^{m}$ 和段落 $P=\left\{w_{t}^{P}\right\}_{t=1}^{n}$  中每个单词的word embedding ( $e_t$)和character embedding( $c_t$ )作为输入，通过Bi-RNN得到对应的编码输出 $u_t$ 。这里的RNN采用的是变体GRU，它的表现与LSTM相当，但是更容易计算。

<center>
$u_{t}^{Q}=\operatorname{BiRNN}_{Q}\left(u_{t-1}^{Q},\left[e_{t}^{Q}, c_{t}^{Q}\right]\right)$
$u_{t}^{P}=\operatorname{BiRNN}_{P}\left(u_{t-1}^{P},\left[e_{t}^{P}, c_{t}^{P}\right]\right)$
</center>

### gated attention-based recurrent networks
这里用经典的attention-based RNN网络的变体将问题信息整合进段落表示，其中额外增加了一个门机制，确定与问题有关的段落中信息的重要性。给定问题和段落表示，通过软对齐问题和段落的单词来生成question-aware段落表示。
<center>
$v_{t}^{P}=\operatorname{RNN}\left(v_{t-1}^{P},\left[u_{t}^{P}, c_{t}\right]\right)$
</center>

这里的 $c_t$ 表示段落中每个词语经过attention机制计算，动态整合了与问题之间的匹配信息的向量。通过以下方式得到：

<center>
$s_{j}^{t}=\mathrm{v}^{\mathrm{T}} \tanh \left(W_{u}^{Q} u_{j}^{Q}+W_{u}^{P} u_{t}^{P}+W_{v}^{P} v_{t-1}^{P}\right)$
$a_{i}^{t}=\exp \left(s_{i}^{t}\right) / \Sigma_{j=1}^{m} \exp \left(s_{j}^{t}\right)$
$c_{t}=\Sigma_{i=1}^{m} a_{i}^{t} u_{i}^{Q}$
</center>

为了确定段落中各部分的重要性，多关注段落中与问题相关的部分，在RNN的输入之前增加了一个门机制：
<center>
$g_{t}=\operatorname{sigmoid}\left(W_{g}\left[u_{t}^{P}, c_{t}\right]\right)$
$\left[u_{t}^{P}, c_{t}\right]^{*}=g_{t} \odot\left[u_{t}^{P}, c_{t}\right]$
</center>

也就是RNN的输入是 $\left[u_{t}^{P}, c_{t}\right]^{*}$

与LSTM或GRU中的门不同，门基于当前通过词及其问题的注意力集中向量，重点关注问题与当前通过词之间的关系。

### self-matching attention

通过基于门控的注意力循环网络，可生成问题意识段落表示$\left\{v_{t}^{P}\right\}_{t=1}^{n}$，以精确定位段落中的重要部分。这种表示的一个问题是它对上下文的了解非常有限。 一个应聘者通常会忽略其周围窗口之外的段落中的重要提示。 而且，在大多数SQuAD数据集中，对于answer的推断必须要使用到Passage ，answer和Passage之间存在某种词汇或句法上的差异。为了解决这个问题，作者建议直接将question-aware passage 表示形式与其自身进行匹配。并提出self-matching attention,动态地收集整个段落的信息给段落当前的词语，把与当前段落词语相关的信息和其匹配的问题信息编码成段落表示：
<center>
$h_{t}^{P}=\operatorname{BiRNN}\left(h_{t-1}^{P},\left[v_{t}^{P}, c_{t}\right]\right)$
</center>
这里的 $c_t$ 是包含了段落当前词语与段落其余词语之间相关性的向量，通过以下方式得到：
<center>
$s_{j}^{t}=\mathrm{v}^{\mathrm{T}} \tanh \left(W_{v}^{P} v_{j}^{P}+W_{v}^{\tilde{P}} v_{t}^{P}\right)$
$a_{i}^{t}=\exp \left(s_{i}^{t}\right) / \Sigma_{j=1}^{n} \exp \left(s_{j}^{t}\right)$
$c_{t}=\Sigma_{i=1}^{n} a_{i}^{t} v_{i}^{P}$
</center>
同样增加一个额外的门来控制RNN的输入。


####  OUTPUT LAYER
我们利用pointer network的变体来预测答案的起始位置 $p^1$ 和结束位置 $p^2$ 。


首先，我们把问题经过attention机制得到的向量作为pointer network的初始隐藏层输入，也就是初始语境信息。这里$h_{t-1}^{a}$表示的是answer recurrent network (pointer network)初始化的隐藏层的状态，计算方式如下：
<center>
$s_{j}^{t}=\mathrm{v}^{\mathrm{T}} \tanh \left(W_{h}^{P} h_{j}^{P}+W_{\mathrm{h}}^{a} h_{t-1}^{a}\right)$
$a_{i}^{t}=\exp \left(s_{i}^{t}\right) / \sum_{j=1}^{n} \exp \left(s_{j}^{t}\right)$

$p^{t}=\operatorname{argmax}\left(a_{1}^{t}, \ldots, a_{n}^{t}\right)$
</center>

当预测的时候，试用$r^Q$ 就是初始语境信息。
<center>
$s_{j}=\mathrm{v}^{\mathrm{T}} \tanh \left(W_{u}^{Q} u_{j}^{Q}+W_{\mathrm{v}}^{Q} V_{r}^{Q}\right)$
$a_{i}=\exp \left(s_{i}\right) / \Sigma_{j=1}^{m} \exp \left(s_{j}\right)$
$r^{Q}=\Sigma_{i=1}^{m} a_{i} u_{i}^{Q}$
</center>

然后，根据给定段落表示，把attention机制作为一个pointer来选取答案在段落中的起始位置，也就是基于初始语境信息，计算段落中每个词语的attention权重，权重最高的作为起始位置：


在得到起始位置之后，利用RNN得到一个新的语境信息，该语境中编码了有关答案起始的信息。新的语境信息计算方式如下：
<center>
$c_{t}=\Sigma_{i=1}^{n} a_{i}^{t} h_{i}^{P}$
$h_{t}^{a}=\operatorname{RNN}\left(h_{t-1}^{a}, c_{t}\right)$
</center>
最后，重复上述起始位置计算方法，利用新的语境信息，就可以计算得到答案的结束位置。

为了训练这个网络，我们需要最小化起始位置ground truth的负对数概率的总和。


## 实验结果分析

### 实验设置细节
本论文主要基于SQuAD来训练及验证模型。SQuAD由100000+个问题构成，随机抽取80%为训练集、10%为开发集、10%为测试集，每个问题的答案都是对应段落中的一个片段。

本文采用Stanford CoreNLP来对每个片段和问题进行预处理。模型中的RNN采用的是其变体GRU。而word embedding则用的是预训练的GloVe embedding，它在训练中不断更新，对于OOV词语用零向量表示。关于character embedding则用一层GRU来训练得到。模型中用3层双向GRU来初步编码问题和段落的信息。模型中后续用到的RNN都是双向的GRU。隐藏层的向量长度是75，用于计算attention权重的长度也是75。每个layer之间的dropout rate是0.2。模型用AdaDelta来进行优化，学习速率是1。

### 基于SQuAD的实验结果
采用两个度量来评估SQuAD: Exact Match（EM）和F1。

Exact match(EM):如果预测答案等于标准答案，则EM得分为1.0，否则为0.0
F1 score:用来计算预测答案和所有标准答案的重合程度的最大值。预测和标准答案可以看作是一个“token袋”，那么token级的F1 score就可以通过计算得到。（precision表示在所有预测结果中预测对的所占比例，recall表示所有预测对的占标准答案的比例）。
下面是基于SQuAD的实验结果：
![-w788](http://roger-markdown.oss-cn-beijing.aliyuncs.com/2020/03/28/15852183884742.jpg)



### 基于MS-MARCO的实验结果
MS-MARCO是另一个关于机器阅读理解的数据集，与SQuAD不同的是每个问题有多个对应的段落，所以本文在实验过程中只是简单的将所有片段按顺序连接在一起。除此之外，MS-MARCO问题的答案并不是段落中连续的片段，所以此处采用的评估方式是BLEU和ROUGE-L。

下面是基于MS-MARCO的实验结果：
![-w1054](http://roger-markdown.oss-cn-beijing.aliyuncs.com/2020/03/28/15852184117595.jpg)



## 讨论
本论文中还讨论了一些可能会提高阅读理解效果的方法，虽然不适用于当前实验的数据集，但是可能在其它数据集中有所帮助：

1. Sentence Ranking：在SQuAD中，一个段落包含几个句子，但是答案总是只1在一个句子中。所以考虑句子的顺序是否可以帮助找到最终的答案。本文做了以下两种尝试来整合句子的排序信息
    1. 训练了一个单独的句子排序模型，将其与答案预测模型相结合；
    2. 将span预测和句子预测作为两个相关的任务，训练一个多任务模型。这两种方法都没有提高最终的实验结果。分析标明句子预测模型的实验结果低于span预测模型，这表明精确的span信息实际上对于选择正确的答案句是至关重要的。
2. Syntax Information：本论文尝试了三种将语法信息融入模型。
    1. 将语法特征作为编码层的输入，这些语法特征包括词性标注、NER、线性的PCFG树标签、依存标签。
    2. 尝试整合一个tree-LSTM模型在编码层之后，使用多输入LSTM在自上而下和自下而上的传递中的依存树路径之后构建隐藏状态。
    3. 用依存分析作为多任务中的一个附加任务。但是以上方法都没有提高实验效果。
3. Multi-hop Inference：尝试增加multi-hop推理模型在pointer network层，但是同样没有提高实验效果。
4. Question Generation：基于数据驱动的方式，利用seq-to-seq的方式生成很多伪数据，给这些伪数据很小的权重，让所有伪数据的权重与真实数据的权重相当。但是由于伪数据的质量不高，所以并没有提高实验结果。


## 结论
本文所提出的R-NET在阅读理解问的数据集SQuAD和MS-MARCO都取得了最好的实验结果。在未来的工作中，将尝试使用语法和知识信息来提高模型效果。此外，还致力于设计新的网络结构来处理需要复杂推理的问题

## 参考
> 
1. [R-NET机器阅读理解（原理解析）](https://zhuanlan.zhihu.com/p/36855204)
2. [《R-NET：MACHINE READING COMPREHENSION》阅读笔记](https://zhuanlan.zhihu.com/p/61502862)
3. [R-NET](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)
4. [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
5. [r-net机器阅读理解实践 code github](https://blog.csdn.net/jdbc/article/details/80657679)
6. [SQUAD的rnet复现踩坑记](https://www.cnblogs.com/rocketfan/p/9103878.html)
7. [R-net的Tensorflow实现：具有自匹配网络的机器阅读理解](https://www.ctolib.com/NLPLearn-R-net.html)
8. [Challenges of reproducing R-NET neural network using Keras](http://yerevann.github.io/2017/08/25/challenges-of-reproducing-r-net-neural-network-using-keras/)