# Fine-Grained Flower Grade Image Classification

## 花朵选取

#### 国家标准



月季切花等级 Grade of rose cut flower



[GB/T 41201-2021](http://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=DDE639E853B2DD324D6987A717DA9D4D)

实施日期：2021-12-31



单头月季切花的开花指数由小到大分 1度 ~ 5度，其中 2度 ~ 4度为适宜采收的开花指数。

可根据不同品种的花瓣数、瓶插后花蕾的开放情况、采收季节及气温、运输距离等进行调整。

花瓣数少、瓶插易开放的品种及夏季气温高时宜采小的开花指数（即早采）

花瓣数多、瓶插不易开放的品种及冬季气温低时宜采大的开花指数（即晚采）

#### 单头月季开花指数划分

| 开花指数/度 | 描述                                                         | 注释                                        |
| :---------: | ------------------------------------------------------------ | ------------------------------------------- |
|      1      | 萼片保持直立，花瓣紧包，未从萼片中伸出，为不适宜采收时期。   | 很难找到，如果没有可以不考虑这个            |
|      2      | 萼片展开 30°~45°，外层花瓣开始松散。适宜夏秋季远距离运输销售。 |                                             |
|      3      | 萼片展开45°~90°，外层花瓣松散展开。适宜冬春季远距离运输。    |                                             |
|      4      | 萼片稍下垂，外层花瓣向外翻卷，多层花瓣展开。适宜冬季近距离运输和就近销售。 |                                             |
|      5      | 花瓣松散，多层花瓣翻卷，花朵露心，不宜采收。                 | 一般不会售卖，可以令234的放置一段时间后获取 |

![grade_detection.png](./presentation/staff/grade_detection.png)







## 待测试的 Model


| Model      | Resolution | Pretrained-weight               | -    |
| ---------- | ---------- | ------------------------------- | ---- |
| Xception   | 229x229    | ImageNet-1k                     | 选用 |
| CCT-7/7x2  | 224x224    | Flowers-102/300 Epochs          |      |
| CCT-14/7x2 | 384x384    | Flowers102/Finetuned/300 Epochs |      |

由于数据集图片过少（简单数据增强之后才只有 300 张左右）

因此，参数量大的模型反而会过拟合，因此不考虑。



## 文件夹结构



e-flower 
 ├── Benchmark.md ==记录实验数据==
 ├── cloud-test ==部署在云服务器的用例==
 │   ├── OxfordFlower.py
 │   └── Xception_model.py
 ├── dataset ==数据集== 因图片过多，目前已传到网盘（PS 阿里云不让分享
 │   ├── Grade_1_1.jpg
 │   └── image_pre_process.ipynb ==预处理==
 ├── data_augmentation ==数据增强== 未做完
 │   ├── 1-5.jpg
 │   ├── 3-3.jpg
 │   ├── Grade_1_1.jpg
 │   ├── Grade_2_1
 │   ├── gray.jpg
 │   ├── image-cut.ipynb
 │   ├── res.jpg
 │   └── transformed.jpg
 ├── deployment ==部署==
 │   ├── EFlowersDataset.py
 │   ├── train.ipynb
 │   └── Xception_model.py
 ├── diagonal-mirro.jpg ==对角翻转的demo==
 ├── docs ==发票 文档等==
 │   ├── Python发票.pdf
 │   ├── 计科杂项.pdf
 │   └── 购花发票.pdf
 ├── MLTarget.md ==目标==
 ├── MyFlowersDataset.py ==Pytorch 的数据集类==
 ├── papers ==参考文献==
 │   ├── 2110.07097.pdf
 │   ├── Compact-Transformer-2104.05704.pdf
 │   ├── electronics-10-02353-v2.pdf
 │   ├── GB_T 41200-2021.pdf
 │   ├── GB_T 41201-2021.pdf
 │   └── Xception-1610.02357.pdf
 ├── presentation ==报告等文档==
 │   ├── 2.0主赛道-创意组-基于机器视觉的农作物蔬菜智能识别系统-马自远-17858417476.pptx
 │   ├── cat_to_name.json
 │   ├── Flower-Representation.pptx
 │   ├── staff
 │   ├── 待完善的参考资料.md
 │   ├── 技术详解.md
 │   └── 鲜切花分析.pdf
 ├── README.md
 └── src ==主要源码==
     ├── fashion_mnist_test.py
     ├── pretrained
     ├── run.py
     ├── utils.py
     └── Xception_model.py
