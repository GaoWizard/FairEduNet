import numpy as np

from aif360.datasets import StructuredDataset


class BinaryLabelDataset(StructuredDataset):
    """所有二分类标签的结构化数据集的基础类."""

    def __init__(self, favorable_label=1., unfavorable_label=0., **kwargs):
        """
        参数:
            favorable_label (float):标签值，认为是有利的标签值 (i.e. "positive").
            unfavorable_label (float):标签值，认为是不被视为有利的标签值 (i.e. "negative").
            **kwargs: 结构化数据集参数.
        """

        # 将正负例标签值存储为属性,转换类型,方便子类继承和使用.
        # 将传入的favorable_label和unfavorable_label参数转换为float类型,并定义为该类的一个属性
        self.favorable_label = float(favorable_label)
        self.unfavorable_label = float(unfavorable_label)

        # 调用父类StructuredDataset的__init__方法, 并传入kwargs参数, 完成父类的初始化工作.
        super(BinaryLabelDataset, self).__init__(**kwargs)

    def validate_dataset(self):
        """错误检查和类型验证

        Raises:
            ValueError: `labels` must be shape [n, 1].
            ValueError: `favorable_label` and `unfavorable_label` must be the
                only values present in `labels`.

        抛出异常：
            值错误： `labels` 必须是 [n, 1] 的形状.
            值错误： `favorable_label` 和 `unfavorable_label` 必须是现在 `labels` 中的唯一值.
                    传入 dataset 的参数 labels 属性中必须只包含 favorable_label 和 unfavorable_label 这两个值。
        """

        # 在验证前修正分数
        # 如果scores和labels值一致,将scores修改为相对于favorable_label的0/1表示.
        if np.all(self.scores == self.labels):
            self.scores = (self.scores == self.favorable_label).astype(np.float64)

        # 调用父类，Dataset类的validate_dataset方法做基础验证.
        super(BinaryLabelDataset, self).validate_dataset()

        # =========================== 检查labels的形状 ===========================
        # 确认labels是不是单列的，如果不是则抛出第1个异常.
        if self.labels.shape[1] != 1:
            raise ValueError("BinaryLabelDataset only supports single-column "
                "labels:\n\tlabels.shape = {}".format(self.labels.shape))

        # =========================== 检查labels中的值 ===========================
        # 使用set判断labels中的值是否只包含favorable_label和unfavorable_label,如果不匹配则抛出ValueError异常.
        if (not set(self.labels.ravel()) <=
                set([self.favorable_label, self.unfavorable_label])):
            raise ValueError("The favorable and unfavorable labels provided do "
                             "not match the labels in the dataset.")
