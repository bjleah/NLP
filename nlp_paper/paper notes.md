**1、论文《End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures》笔记：**<br />
&emsp;&emsp;**（1）模型结构图：**<br />
![image](https://img2022.cnblogs.com/blog/2603071/202206/2603071-20220606113924797-922455141.png)<br />
&emsp;&emsp;**（2）模型结构说明：** 两个LSTM层，第一个bi-LSTM进行实体识别，第二个Tree-LSTM进行关系分类，整个模型依托于依存标注数据。<br />
&emsp;&emsp;**（3）要点：** 关于Tree-LSTM，每个矩形为lstm单元，根据依存结构树结构来进行数据的先后计算。<br />
&emsp;&emsp;**（4）模糊点：** 一般语法树结构当前词被支配词只有一个，但这里有多个？<br />
&emsp;&emsp;**（5）依存结构树例：**<br />
![image](https://img2022.cnblogs.com/blog/2603071/202206/2603071-20220606114528953-880470618.webp)<br />
&emsp;&emsp;根据支配与被支配关系建立后，就当前LSTM需要将其倒置，然后根据先后顺序计算。<br />
&emsp;&emsp;**（6）模型两个任务：** 任务一：基于bi-lstm的实体识别（序列标注：BIL(last)OU）；关系分类-两个L标签的实体或U标签实体（双向LSTM树结构）；<br />
&emsp;&emsp;&emsp;**任务一：** 数据输入：词性embedding+词embedding，输出：实体识别标注序列<br />
&emsp;&emsp;&emsp;**任务二：** 数据输入：实体1的embedding（任务一的sequence输出）+ 依存关系类别embedding +对应实体预测得到的标签embedding<br />
&emsp;&emsp;&emsp;**目标词对儿的三种表示结构：** 1）SPTree（最短路径结构)；2）SubTree（最近公共祖先）；3）Full-Tree（整棵依存树）one node type:采用SPTree<br />
&emsp;&emsp;&emsp;1）对于两个带有L或U标签（BILOU schema）的词，可以构建一个候选的关系<br />
&emsp;&emsp;&emsp;2）NN为每一个候选关系预测一个关系标签，并带有方向（除了负关系外，negative relation）<br />

**2、论文《Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme》笔记：** <br />
&emsp;&emsp;**(1)模型结构图**<br />
![image](https://img2022.cnblogs.com/blog/2603071/202206/2603071-20220606174709564-1805285240.png)
&emsp;&emsp;**(2)模型结构说明**<br />
&emsp;&emsp;编码：双向lstm进行序列编码 <br />
&emsp;&emsp;解码：改进lstm解码（LSTMd）<br />
&emsp;&emsp;预测标签：<br />
![image](https://img2022.cnblogs.com/blog/2603071/202206/2603071-20220606180826703-1240490797.png)

&emsp;&emsp;(1) 实体中词的位置信息 { B，I，E，S，O } 分别表示{实体开始，实体内部，实体结束，单个实体，无关词}；<br />
&emsp;&emsp;(2) 实体关系类型信息，需根据关系类型进行标记，分为多个类别，如 { CF，CP，… } ；<br />
&emsp;&emsp;(3) 实体角色信息 { 1，2 } 分别表示 { 实体 1，实体 2 }<br />

&emsp;&emsp;**(3)缺点**<br />
&emsp;&emsp;不能识别重叠的实体关系<br />
