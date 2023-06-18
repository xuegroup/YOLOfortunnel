# coding utf-8
# 作者：贾非
# 时间：2023/3/7 14:33
"""
使用顺序
1. image_prepare：切割图像、拼接图像、翻转图像、移动图像
2. dat_cal：统计数据集信息，检验标签，生成类别json文件
3. xml2yolo：将xml格式转为yolo并保存，生成训练集图像与测试集图像绝对路径
"""


def parse_xml_to_dict(xml_doc):
    """
    :param xml_doc: 利用read打开后读取得到str文件，再由etree转换为xml格式传入
    :return: 字典形式的xml内容
    """
    if len(xml_doc) == 0:
        return {xml_doc.tag: xml_doc.text}  # 遍历至底层，获取每个tag对应内容

    result = {}
    for child in xml_doc:
        child_res = parse_xml_to_dict(child)  # 嵌套遍历获取每个tag对应的内容，套娃
        if child.tag != 'object':
            result[child.tag] = child_res[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []  # 可能存在多个object，也就是标注框
            result[child.tag].append(child_res[child.tag])
    return {xml_doc.tag: result}
