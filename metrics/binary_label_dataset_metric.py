# 二分类数据集的各项指标

import numpy as np
from sklearn.neighbors import NearestNeighbors
from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor
from aif360.datasets import BinaryLabelDataset
from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset
from aif360.metrics import DatasetMetric, utils
from aif360.algorithms.inprocessing.gerryfair.clean import *


class BinaryLabelDatasetMetric(DatasetMetric):
    # 基于单个目标（aif360.datasets.BinaryLabelDataset）的计算指标类
    """Class for computing metrics based on a single obj:`~aif360.datasets.BinaryLabelDataset`.
    """

    def __init__(self, dataset, unprivileged_groups=None, privileged_groups=None):
        """
        参数，Args:
            
            dataset (BinaryLabelDataset): 一个二进制标签数据集。
            
            privileged_groups (list(dict)): 受保护组。格式是一个'dicts'的列表，其中键是'protected_attribute_names'（保护属性的姓名），值是
            'protected_attributes'中的值。每个'dict'元素描述一个组。更多细节描述参考案例。
            
            unprivileged_groups (list(dict)): 非受保护组，和'privileged_groups'格式相同。

       Raises:      
            TypeError: `dataset` must be a obj:`~aif360.datasets.BinaryLabelDataset` type.
       
       抛出异常:
            类型错误：'dataset'必须是'aif360.datasets.BinaryLabelDataset'（二分类标签数据集）类型
        """
        
        # dataset不是BinaryLabelDataset,并且也不是MulticlassLabelDataset时，报错：
        if not isinstance(dataset, BinaryLabelDataset) and not isinstance(dataset, MulticlassLabelDataset) :
            raise TypeError("'dataset' should be a BinaryLabelDataset or a MulticlassLabelDataset")


        """
        super(BinaryLabelDatasetMetric, self).__init__(...) 代码，调用了 DatasetMetric 类的 __init__ 方法，并传递了 dataset、unprivileged_groups 
        和 privileged_groups 作为参数。
        简而言之，super() 函数在这里确保了子类 BinaryLabelDatasetMetric 在初始化时也执行了父类 DatasetMetric 的初始化逻辑。
        """
        # sets self.dataset, self.unprivileged_groups, self.privileged_groups
        super(BinaryLabelDatasetMetric, self).__init__(dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        # dataset是MulticlassLabelDataset时：
        if isinstance(dataset, MulticlassLabelDataset):
            # 受保护属性为1，非受保护属性为0
            fav_label_value = 1.
            unfav_label_value = 0.

            self.dataset = self.dataset.copy()

            # 找到所有含有受保护属性的标签
            
            # np.equal.outer：进行比较，得到受保护属性标签
            
            # np.logical_or.reduce： 沿着指定的轴（默认为轴 0）应用逻辑或操作，将多个布尔值合并为一个布尔值。
            #  如果 self.dataset.favorable_label 中的任何一个标签与 self.dataset.labels 中的某个标签相等，那么结果数组的相应位置就会是 True。
            
            # fav_idx:布尔数组，其长度与 self.dataset.labels 相同。
            #  对于 self.dataset.labels 中的每一个标签，如果它与 self.dataset.favorable_label 中的任何一个标签相等，那么 fav_idx 的相应位置就会是
            #  True，否则是 False。
            fav_idx = np.logical_or.reduce(np.equal.outer(self.dataset.favorable_label, self.dataset.labels))
            # 给对应位置的值打标签，True为favorable_label，False为unfav_label_value
            self.dataset.labels = np.where(fav_idx, fav_label_value, unfav_label_value)
            # 转换为float格式
            self.dataset.favorable_label = float(fav_label_value)
            self.dataset.unfavorable_label = float(unfav_label_value)

    # 定义num_positives：正样本数量
    def num_positives(self, privileged=None):
        r"""计算正例的数量，可以选择以受保护属性为条件

        参数:
            privileged (bool，额外可选项): 布尔值，规定是否将度量指标设置为"privileged_groups"
            如果为 "true"，则规定是以 "privileged_groups"（受保护属性组）为条件，如果是"false"，则是以 "unrivileged_groups"（非特权组）为条件。
            默认是"None"，表示度量指标是在整个数据集上进行计算的。

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups` must be
                must be provided at initialization to condition on them.
                
        抛出异常：
            属性错误，`privileged_groups` 或 `unprivileged_groups`必须在初始化时提供，以便进行条件化 
        """
        # 调用 _to_condition 方法，传递参数privileged，详情见dataset_metric.py中的定义
        # 返回一个条件，该条件用于确定我们是否只考虑受保护的群体或非受保护的群体。
        condition = self._to_condition(privileged)
        # 调用 compute_num_pos_neg 方法，详见utils.py中的定义
        # 计算正例或负例的数量，具体取决于传递给它的参数。
        return utils.compute_num_pos_neg(self.dataset.protected_attributes,
            self.dataset.labels, self.dataset.instance_weights,
            self.dataset.protected_attribute_names,
            self.dataset.favorable_label, condition=condition)

    # 定义num_negatives：负样本数量
    def num_negatives(self, privileged=None):
        r"""计算负例的数量

        参数:
            privileged (bool，额外可选项): 布尔值，规定是否将度量指标设置为"privileged_groups"
            如果为 "true"，则规定是以 "privileged_groups"（受保护属性组）为条件，如果是"false"，则是以 "unrivileged_groups"（非特权组）为条件。
            默认是"None"，表示度量指标是在整个数据集上进行计算的。

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups` must be
                must be provided at initialization to condition on them.
       
       抛出异常：
           属性错误，`privileged_groups` 或 `unprivileged_groups`必须在初始化时提供，以便进行条件化 
        """
        # 返回一个条件，该条件用于确定我们是否只考虑受保护的群体或非受保护的群体。
        condition = self._to_condition(privileged)
        # 计算正例或负例的数量，具体取决于传递给它的参数。
        # 正样本数量中的参数self.dataset.favorable_label，在负样本数量中变为self.dataset.unfavorable_label
        return utils.compute_num_pos_neg(self.dataset.protected_attributes,
            self.dataset.labels, self.dataset.instance_weights,
            self.dataset.protected_attribute_names,
            self.dataset.unfavorable_label, condition=condition)

    def base_rate(self, privileged=None):
        """计算base rate,正率的比例

        参数:
            privileged (bool, 可选项): 布尔值，规定是否将度量指标设置为"privileged_groups"
            如果为 "true"，则规定是以 "privileged_groups"（受受保护属性组）为条件，如果是"false"，则是以 "unrivileged_groups"（非受保护属性组）为条件。
            默认是"None"，表示度量指标是在整个数据集上进行计算的。

        返回值:
            float，浮点类型: Base rate，基准率 (可选条件项).
        """
        return (self.num_positives(privileged=privileged)
              / self.num_instances(privileged=privileged))

    def disparate_impact(self):
        r"""
        计算Disparate impact，差异影响
        """
        return self.ratio(self.base_rate)

    def statistical_parity_difference(self):
        r"""
        计算Statistical Parity Difference统计学均等差异
        .. math::
           Pr(Y = 1 | D = \text{unprivileged}) - Pr(Y = 1 | D = \text{privileged})
        """
        return self.difference(self.base_rate)

    def consistency(self, n_neighbors=5):
        r"""
        计算consistency，一致性
        来自参考文献[1]中的个体公平性测量指标，用于衡量相似实例的标签的相似程度

        参数:
            n_neighbors (int, 可选项): knn 计算中的邻居数量
        """

        
        # X：数据集的特征
        X = self.dataset.features
        # num_samples：数据集中的实例数量
        num_samples = X.shape[0]
        # y:数据集中的标签
        y = self.dataset.labels

        # 训练KNN网络
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        nbrs.fit(X)
        _, indices = nbrs.kneighbors(X)

        # 计算一致性分数
        consistency = 0.0
        for i in range(num_samples):
            # 计算了 数据点标签 和 邻居标签 的差值 的绝对值之和
            consistency += np.abs(y[i] - np.mean(y[indices[i]]))
        #一致性分数 = 1- 计算所有数据点的平均差异，数值越高越公平
        consistency = 1.0 - consistency/num_samples

        return consistency


    def _smoothed_base_rates(self, labels, concentration=1.0):
        """
        计算 Smoothed base rates，平滑正率的比例
        计算了 数据集中每个交叉组的 Dirichlet-平滑正率比例。
        """
        # Dirichlet-平滑的参数
        if concentration < 0:
            raise ValueError("Concentration parameter must be non-negative.")
        # 二分类数据集
        num_classes = 2
        # dirichlet_alpha： Dirichlet-平滑的参数，它等于concentration除以类别数。
        dirichlet_alpha = concentration / num_classes

        # 计算所有交叉组（多个敏感属性标签的组）的数量，如black-women, white-man, 等等
        # 去除重复行，获得所有受保护属性的列表
        intersect_groups = np.unique(self.dataset.protected_attributes, axis=0)
        # num_intersects：交叉组总的数量
        num_intersects = len(intersect_groups)
        # counts_pos：正例数量，初始化
        counts_pos = np.zeros(num_intersects)
        # counts_total：总数量，初始化
        counts_total = np.zeros(num_intersects)
        # 遍历每个交叉组
        for i in range(num_intersects):
            # 将交叉组的属性名称和值配对，并转换为字典dict形式，放入一个列表中作为condition
            condition = [dict(zip(self.dataset.protected_attribute_names,
                                  intersect_groups[i]))]
            # 交叉组的总数量
            counts_total[i] = utils.compute_num_instances(
                    self.dataset.protected_attributes,
                    self.dataset.instance_weights,
                    self.dataset.protected_attribute_names, condition=condition)
            # 交叉组的正例数量
            counts_pos[i] = utils.compute_num_pos_neg(
                    self.dataset.protected_attributes, labels,
                    self.dataset.instance_weights,
                    self.dataset.protected_attribute_names,
                    self.dataset.favorable_label, condition=condition)

        # 给定S时的Y的概率(p(y=1|S))
        # probability of y given S (p(y=1|S))
        return (counts_pos + dirichlet_alpha) / (counts_total + concentration)

    def smoothed_empirical_differential_fairness(self, concentration=1.0):
        """平滑经验差分公平性，Smoothed EDF [2]

        参数:
            concentration (float, optional): 浓缩度参数，用于 Dirichlet-平滑
            必须是非负数

        案例:
            为了使用在非二元属性上，列 (column) 必须转换为序数列 (ordinal).

            >>> mapping = {'Black': 0, 'White': 1, 'Asian-Pac-Islander': 2,
            ... 'Amer-Indian-Eskimo': 3, 'Other': 4}
            >>> def map_race(df):
            ...     df['race-num'] = df.race.map(mapping)
            ...     return df
            ...
            >>> adult = AdultDataset(protected_attribute_names=['sex',
            ... 'race-num'], privileged_classes=[['Male'], [1]],
            ... categorical_features=['workclass', 'education',
            ... 'marital-status', 'occupation', 'relationship',
            ... 'native-country', 'race'], custom_preprocessing=map_race)
            >>> metric = BinaryLabelDatasetMetric(adult)
            >>> metric.smoothed_empirical_differential_fairness()
            1.7547611985549287

        参考文献:
             [2] J. R. Foulds, R. Islam, K. N. Keya, and S. Pan,"An Intersectional Definition of Fairness," arXiv preprint
             ,arXiv:1807.08362, 2018.
        """
        # 计算平滑正例比率
        sbr = self._smoothed_base_rates(self.dataset.labels, concentration)

        # 计算两个交叉组之间的正样本比例的差异
        def pos_ratio(i, j):
            return abs(np.log(sbr[i]) - np.log(sbr[j]))

        # 计算两个交叉组之间的负样本比例的差异
        def neg_ratio(i, j):
            return abs(np.log(1 - sbr[i]) - np.log(1 - sbr[j]))

        # 计算总的差异公平性
        # 从所有的交叉组对中找到最大的差异公平性
        # 对于每一对交叉组，这个 max 函数计算了正样本和负样本的比例差异，并取其中的最大值。
        return max(max(pos_ratio(i, j), neg_ratio(i, j))
                   for i in range(len(sbr)) for j in range(len(sbr)) if i != j)

    # ============================== ALIASES 其他指标补充 ===================================
    def mean_difference(self):
        """statistical_parity_difference 方法的别名 """
        return self.statistical_parity_difference()


    def rich_subgroup(self, predictions, fairness_def='FP'):
        """
        Audlt 数据集中，根据敏感属性的线性阈值，通过rich subgroups，丰富子群的定义进行审计

            参数： fairness_def 是关于假阳性 (false positive) 或假阴性 (false negative) 的丰富子群的 'FP' 或 'FN'。
                  prediction 是 可哈希的 预测元组。通常是 GerryFairClassifier 的 labels 属性


            返回值: 关于 fairness_def 的 gamma disparity

            案例: see examples/demo_gerryfair.ipynb
        """

        auditor = Auditor(self.dataset, fairness_def)

        # 变成可哈希的类型
        y = array_to_tuple(self.dataset.labels)
        predictions = array_to_tuple(predictions)

        # 如果是'FP(假阳性)',就返回平均值( y=0 时的预测值)
        # 如果是'FN(假阴性)',就返回平均值( y=1 时的预测值)
        metric_baseline = auditor.get_baseline(y, predictions)

        # 返回有最大差异(the largest disparity)的属性组
        group = auditor.get_group(predictions, metric_baseline)

        return group.weighted_disparity

