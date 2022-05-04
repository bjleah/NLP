+ ##### 此模块利用中文的预训练模型，预测两个文本的相似度，在语义方面得到结果
+ ##### 此模块采用了百度的预训练模型ernie作为基础模型，采用了其经典的gram模型，模型结构以bert为基础，在预训练阶段增加了短语和实体层面的预测，并且用了许多优质的语料。
+ ##### 各模块的作用如下：
+ data：存储了finetune的语料
+ ernie: 为ernie的源代码的核心的模块
+ model-ernie-gram-zh.1:预训练的模型
+ propeller: 常用的数据处理接口
+ ckpt.bin: finetune后的模型参数文件
+ example : 使用ernie的一小段代码
+ get_param: 定义参数解析模块
+ model_predict:加载模型，并预测文件形式的数据和单条数据
