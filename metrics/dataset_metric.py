import copy

import numpy as np

from aif360.datasets import StructuredDataset
from aif360.metrics import Metric, utils


class DatasetMetric(Metric):
    """基于结构化的数据集(StructuredDataset)的计算指标类"""

    def __init__(self, dataset, unprivileged_groups=None, privileged_groups=None):
        """
        参数:
            dataset (StructuredDataset): 一个结构化的数据集
            privileged_groups (list(dict)): 受保护属性组。形式为一个`dicts`的列表，其中键是`protected_attribute_names`,
                值是`protected_attributes`中的值.每个`dicts`元素都描述了一个单独的组，

            unprivileged_groups (list(dict)): 非受保护属性组，形式和'privileged_groups'相同.

        Raises:
            TypeError: `dataset` must be a
                :obj:`~aif360.datasets.StructuredDataset` type.
            ValueError: `privileged_groups` and `unprivileged_groups` must be
                disjoint.

        抛出异常：
            类型错误：`dataset`必须是一个`~aif360.datasets.StructuredDataset`类型.
            数值错误：`privileged_groups` and `unprivileged_groups` 必须是互斥的(disjoint).

        案例:
            >>> from aif360.datasets import GermanDataset
            >>> german = GermanDataset()
            >>> u = [{'sex': 1, 'age': 1}, {'sex': 0}]
            >>> p = [{'sex': 1, 'age': 0}]
            >>> dm = DatasetMetric(german, unprivileged_groups=u, privileged_groups=p)
        """
        if not isinstance(dataset, StructuredDataset):
            raise TypeError("'dataset' should be a StructuredDataset")

        # sets self.dataset
        super(DatasetMetric, self).__init__(dataset)

        # TODO: 研究一下这里是不是应该使用深拷贝，deepcopy?
        # self.privileged_groups = privileged_groups
        self.privileged_groups = copy.deepcopy(privileged_groups)
        # self.unprivileged_groups = unprivileged_groups
        self.unprivileged_groups = copy.deepcopy(unprivileged_groups)

        # 如果未提供数据集的参数`privileged_groups`和`unprivileged_groups`为空，则跳过数值检查
        if not self.privileged_groups or not self.unprivileged_groups:
            return

        # 使用utils.compute_boolean_conditioning_vector根据privileged_groups生成一个priv_mask
        # 这个mask表明哪些样本属于privileged_groups.同理,用unprivileged_groups生成unpriv_mask。
        priv_mask = utils.compute_boolean_conditioning_vector(
            self.dataset.protected_attributes,
            self.dataset.protected_attribute_names, self.privileged_groups)
        unpriv_mask = utils.compute_boolean_conditioning_vector(
            self.dataset.protected_attributes,
            self.dataset.protected_attribute_names, self.unprivileged_groups)
        # 将两个mask做逻辑与操作(np.logical_and),得到的是两个组合在一起的样本
        # 如果两个组合在一起的样本不为空(np.any返回True),说明两个组有交集,就报错,要求两个组必须互斥(disjoint).
        if np.any(np.logical_and(priv_mask, unpriv_mask)):
            raise ValueError("'privileged_groups' and 'unprivileged_groups'"
                             " must be disjoint.")

    def _to_condition(self, privileged):
        """将一个布尔条件，转换为用于创建条件向量的分组指定格式.
           这个方法将传入的布尔 privileged 参数转换成 privileged_groups 或 unprivileged_groups 的格式,
           后续可以用它来构造条件向量,从而选取不同的样本子集进行度量计算。

            布尔条件(boolean condition):指传入的 privileged 参数,它是一个布尔值.
            条件向量(conditioning vector):指根据分组方式创建的一个向量,用 1 和 0 表示样本是否属于某组.
            分组指定格式(group-specifying format):指 privileged_groups 或 unprivileged_groups,它们表示样本的分组方式。
        """

        # 如果privileged为True,但在初始化时没有提供privileged_groups,抛出AttributeError异常.
        if privileged is True and self.privileged_groups is None:
            raise AttributeError("'privileged_groups' was not provided when "
                                 "this object was initialized.")
        if privileged is False and self.unprivileged_groups is None:
            raise AttributeError("'unprivileged_groups' was not provided when "
                                 "this object was initialized.")

        # 如果privileged为None,直接返回None.
        if privileged is None:
            return None
        # 如果privileged为True,返回privileged_groups. 如果privileged为False,返回unprivileged_groups.
        return self.privileged_groups if privileged else self.unprivileged_groups

    def difference(self, metric_fun):
        """计算非有利和有利的组的度量指标的差异
        """
        return metric_fun(privileged=False) - metric_fun(privileged=True)

    def ratio(self, metric_fun):
        """一个通用的比率计算接口,可以计算受保护属性组和受保护属性组的任意度量指标的比值
        metric_fun: 一个可调用的函数(callable),表示一个要计算的度量函数.
        """
        return metric_fun(privileged=False) / metric_fun(privileged=True)

    def num_instances(self, privileged=None):
        """计算实例的数量，数学语言：`n`，数据集中的实例数，必要时可以以受保护的属性为条件.

        参数:
            privileged (bool, 额外可选项): 布尔值，规定此度量指标的一个条件.
                如果此值为`True`，以`privileged_groups`为条件计算, 此值为 `False`，以`unprivileged_groups`为条件计算.
                默认为 `None`，表面该指标是在整个数据集上计算.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.

        抛出异常：
            属性错误：`privileged_groups` 或 `unprivileged_groups` 必须是在初始化时提供，以便对其进行条件设置.
        """

        condition = self._to_condition(privileged)
        return utils.compute_num_instances(self.dataset.protected_attributes,
            self.dataset.instance_weights,
            self.dataset.protected_attribute_names, condition=condition)
