**1、论文《End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures》笔记：**<br />
&emsp;&emsp;**（1）模型结构图：**<br />
![image](https://img2022.cnblogs.com/blog/2603071/202206/2603071-20220606113924797-922455141.png)<br />
&emsp;&emsp;**（2）模型结构说明：** 两个LSTM层，第一个bi-LSTM进行实体识别，第二个Tree-LSTM进行关系分类，整个模型依托于依存标注数据。<br />
&emsp;&emsp;**（3）要点：** 关于Tree-LSTM，每个矩形为lstm单元，根据依存结构树结构来进行数据的先后计算。<br />
&emsp;&emsp;**（4）模糊点：** 一般语法树结构当前词被支配词只有一个，但这里有多个？<br />
&emsp;&emsp;**（5）依存结构树例：**<br />
![image](https://img2022.cnblogs.com/blog/2603071/202206/2603071-20220606114528953-880470618.webp)<br />
&emsp;&emsp;根据支配与被支配关系建立后，就当前LSTM需要将其倒置，然后根据先后顺序计算。<br />
