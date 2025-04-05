from logging import warning

import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset


class StandardDataset(BinaryLabelDataset):
    """ AIF360包提供的每个`BinaryLabelDataset`对象(:obj:`BinaryLabelDataset`)的基类

    添加自定义数据集时，严格来说没有必要继承此类，但此基类可能很有用

    这个类的代码非常宽松(loosely)的基于
    https://github.com/algofairness/fairness-comparison.
    """

    def __init__(self, df, label_name, favorable_classes,
                 protected_attribute_names, privileged_classes,
                 instance_weights_name='', scores_name='',
                 categorical_features=[], features_to_keep=[],
                 features_to_drop=[], na_values=[], custom_preprocessing=None,
                 metadata=None):
        """
        在调用 `super().__init__` 前，StandardDataset的子类应当执行以下操作:

            1. 从原始文件(raw file)加载 dataframe

        然后，这个类将会经历一个标准的数据预处理过程，其中包括:

            2.(可选项) 执行一些特定于数据集的预处理(例如，重命名 列/值, 处理缺失值)

            3. 删除不需要的列 (可看features_to_keep` 和 `features_to_drop` 了解更多细节)

            4. 删除缺失值的行.

            5. 为分类变量创建 one-hot 编码

            6. 给受保护属性映射二分类的特权/非特权值(1/0)

            7. 给标签映射二分类的有利的/非有利的标签(1/0)

        参数:
            df (pandas.DataFrame): DataFrame格式的数据,在其上执行标准处理.
            label_name: `df`中标签(label)列的名称.
            favorable_classes (列表(list) 或 函数(function)):
                被视为有利的(favorable)标签值 或 一个布尔值函数，如果有利则返回 `True`，其他的是非有利的.
                标签值如果不是二进制类型和数值类型,会被映射成 1(有利的, favorable) 和 0 (非有利的, unfavorable)
            protected_attribute_names (list): 与`df`中的受保护属性列对应的名称的列表(list).
            privileged_classes (list(列表(list) 或 函数(function)):
                每一个元素都是一个列表.这个列表被视为特权的组，或者是一个布尔函数，
                当`protected_attribute_names`中的相应列有特权，就返回`True`,其他的则是非特权.
                如果这些值不是数值类型，则这些值将会被映射为 1(特权的，privileged) 和 0(非特权的，unprivileged).
            instance_weights_name (可选项): `df` 中实例权重列的名称.
            categorical_features (可选项, list): DataFrame中要拓展为 one-hot编码 的列名称的列表
            features_to_keep (可选项, list): 要保留的列名. 除了出现在 `protected_attribute_names`,
                `categorical_features`, `label_name` 或 `instance_weights_name`,其他列都删除.如果未提供，默认是保留所有列.
            features_to_drop (可选项, list): 要删除的列.
                *注意: 这个参数覆盖 * `features_to_keep`.
            na_values (可选项): 识别为NA的其他额外字符串.可看函数 :func:`pandas.read_csv` 了解更多细节.
            custom_preprocessing (function):作用于并返回于 DataFrame 的函数对象.(f: DataFrame -> DataFrame).
                                            如果是 `None`, 则不应用额外的预处理.
            metadata (可选项): 额外附加的元数据 (metadata).
        """
        # 2. 执行特定数据集的预处理
        if custom_preprocessing:
            df = custom_preprocessing(df)

        # 3. 删除不需要的列
        # 默认features_to_keep是保留所有列
        features_to_keep = features_to_keep or df.columns.tolist()
        # keep是四个list参数的并集
        keep = (set(features_to_keep) | set(protected_attribute_names)
              | set(categorical_features) | set([label_name]))
        # 如果存在instance_weights_name中，加入到keep中.
        if instance_weights_name:
            keep |= set([instance_weights_name])
        # keep的列基础上,删除features_to_drop中指定的列,根据列在df中的位置进行排序.
        df = df[sorted(keep - set(features_to_drop), key=df.columns.get_loc)]
        # categorical_features 执行相同的处理.
        categorical_features = sorted(set(categorical_features) - set(features_to_drop), key=df.columns.get_loc)

        # 4. 删除任何有缺失值的行.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        if count > 0:
            warning("Missing Data: {} rows removed from {}.".format(count,
                    type(self).__name__))
        df = dropped

        # 5. 为分类变量(categorical variables)列创建 one-hot 编码,分隔符是"="
        df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')

        # 6. 给受保护属性(protected attributes)映射二分类的特权/非特权值(1/0)
        # 用于存储映射结果的privileged_protected_attributes、unprivileged_protected_attributes list列表
        privileged_protected_attributes = []
        unprivileged_protected_attributes = []
        for attr, vals in zip(protected_attribute_names, privileged_classes):
            # 特权值是1,非特权值是0.
            privileged_values = [1.]
            unprivileged_values = [0.]
            # 检查值是否能够可调用对象(callable),即函数
            if callable(vals):
                # 通过apply将该函数应用于df[attr]这一列,用来实现自定义的映射逻辑.
                df[attr] = df[attr].apply(vals)
            # 不是callable类型，检查df[attr]是否是数值类型;issubdtype用于检查DataFrame列的数据类型.
            elif np.issubdtype(df[attr].dtype, np.number):
                # 这个属性是数值类型;不需要重新映射.
                privileged_values = vals
                # 得到df中attr列除特权值(vals)之外的所有唯一值,并转换为列表
                unprivileged_values = list(set(df[attr]).difference(vals))
            # 不是callable类型也不是数值,则进行 0/1 重新映射.
            else:
                # 找到所有能匹配任意属性值的实例.
                # 判断df[attr]中的每个值与vals是否相等，返回一个比较结果的布尔类型的二维数组
                # 逻辑或运算，并聚合成一维数组
                priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                # df.loc通过布尔索引定位到priv为True的行,将其attr列的值设置为1.
                df.loc[priv, attr] = privileged_values[0]
                # ~priv为取反的priv,对应非特权值位置.将非特权值位置的attr设为0。
                df.loc[~priv, attr] = unprivileged_values[0]

            # 添加对应的存储结果，并转换为np.float64类型
            privileged_protected_attributes.append(
                np.array(privileged_values, dtype=np.float64))
            unprivileged_protected_attributes.append(
                np.array(unprivileged_values, dtype=np.float64))

        # 7. 给标签映射二分类标签(1/0)
        favorable_label = 1.
        unfavorable_label = 0.
        if callable(favorable_classes):
            df[label_name] = df[label_name].apply(favorable_classes)
        # 是否是数值类型，是否只有两个不同的值
        elif np.issubdtype(df[label_name], np.number) and len(set(df[label_name])) == 2:
            # 标签已经是二分类;无需修改
            # 接取标签列中两个不同值,作为favorable_label和unfavorable_label.
            favorable_label = favorable_classes[0]
            unfavorable_label = set(df[label_name]).difference(favorable_classes).pop()
        else:
            # 到所有能匹配任意有利结果类的实例
            pos = np.logical_or.reduce(np.equal.outer(favorable_classes, 
                                                      df[label_name].to_numpy()))
            df.loc[pos, label_name] = favorable_label
            df.loc[~pos, label_name] = unfavorable_label

        # OOP 设计模式,通过继承和多态实现代码复用和可维护性.
        # 将处理后的dataframe等数据传给父类,并调用父类BinaryLabelDataset的构造函数__init__来完成数据集的初始化
        super(StandardDataset, self).__init__(df=df, label_names=[label_name],
            protected_attribute_names=protected_attribute_names,
            privileged_protected_attributes=privileged_protected_attributes,
            unprivileged_protected_attributes=unprivileged_protected_attributes,
            instance_weights_name=instance_weights_name,
            scores_names=[scores_name] if scores_name else [],
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label, metadata=metadata)
