# coding utf-8
# 作者：贾非
# 时间：2023/3/16 13:55
"""
模型搭建主要基于cfg/yoloxx.yaml中文件进行搭建
1. layers中包含了模型搭建中的各种基本层，如Conv, C3, SPPF, Concat等，与yaml文件中对应，读取yaml文件时，利用eval函数实现计算
2. parse_model_cfg实现模型的搭建，主要目的是明确搭建过程中out_channels，以及Concat所需要融合的特征图索引
   yolo中主要实现2中数据的传输运算，其中包含了Detect基本层，为方便理解，将其放在了yolo中
与layers中不同的是，Detect需要计算推理阶段的结果，将预测结果反算至原图，因此需要生成grid，并为每个grid point放置anchor
"""