import os

import pandas as pd

from aif360.datasets import StandardDataset

# 标签的默认映射
default_mappings = {
    'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'},
                                 {1.0: 'Male', 0.0: 'Female'}]
}

class AdultDataset(StandardDataset):
    """成年人口普查收入数据集, Adult Census Income Dataset.

    详见: file:`aif360/data/raw/adult/README.md`.
    """

    def __init__(self, label_name='income-per-year',
                 favorable_classes=['>50K', '>50K.'],
                 protected_attribute_names=['race', 'sex'],
                 privileged_classes=[['White'], ['Male']],
                 instance_weights_name=None,
                 categorical_features=['workclass', 'education',
                     'marital-status', 'occupation', 'relationship',
                     'native-country'],
                 features_to_keep=[], features_to_drop=['fnlwgt'],
                 na_values=['?'], custom_preprocessing=None,
                 metadata=default_mappings):
        """ 有关参数的说明，详见 :obj:`StandardDataset`。

        案例:
            下面将实例化一个使用 `fnlwgt` 来调整实例权重特征的数据集.

            >>> from aif360.datasets import AdultDataset
            >>> ad = AdultDataset(instance_weights_name='fnlwgt',
            ... features_to_drop=[])
            WARNING:root:Missing Data: 3620 rows removed from dataset.
            # 测试实例的权重是否不全为 1
            >>> not np.all(ad.instance_weights == 1.)
            True

            实例化一个只使用数值型(numerical)特征和一个受保护属性变得数据集,运行以下代码:

            >>> single_protected = ['sex']
            >>> single_privileged = [['Male']]
            >>> ad = AdultDataset(protected_attribute_names=single_protected,
            ... privileged_classes=single_privileged,
            ... categorical_features=[],
            ... features_to_keep=['age', 'education-num'])
            >>> print(ad.feature_names)
            ['education-num', 'age', 'sex']
            >>> print(ad.label_names)
            ['income-per-year']

            注意: `protected_attribute_names` 和 `label_name` 会被保留，即使他们不在 `features_to_keep` 中给出.

            在某些情况下，追踪 `float -> str` 的保护属性和/或标签 (a mapping from
            `float -> str` for protected attributes and/or labels) 可能会很有用.
            如果我使用的案例和默认情况不同，我可以修改存储在`metadata`中的映射:

            >>> label_map = {1.0: '>50K', 0.0: '<=50K'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> ad = AdultDataset(protected_attribute_names=['sex'],
            ... categorical_features=['workclass', 'education', 'marital-status',
            ... 'occupation', 'relationship', 'native-country', 'race'],
            ... privileged_classes=[['Male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            注意现在在 `categorical_features` 中,我添加了`race`特征.
            现在数据集就会保留这个信息，并能用于更具描述性的可视化.
        """

        # 构造训练集的训练路径train_path和测试路径test_path
        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'data', 'raw', 'adult', 'adult.data')
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'data', 'raw', 'adult', 'adult.test')
        # 根据 adult.names 定义列名
        column_names = ['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year']
        try:
            # 加载训练集和测试集
            train = pd.read_csv(train_path, header=None, names=column_names,
                skipinitialspace=True, na_values=na_values)
            test = pd.read_csv(test_path, header=0, names=column_names,
                skipinitialspace=True, na_values=na_values)
        except IOError as err:
            # # 如果文件加载错误,打印错误并给出下载提示
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'adult'))))
            import sys
            sys.exit(1)

        # 将训练集和测试集concat,合并为一个DataFrame
        df = pd.concat([test, train], ignore_index=True)
        # 调用父类StandardDataset的初始化函数,传入合并后的数据集df以及其他参数
        # StandardDataset的初始化函数会处理这些参数,对数据集进行处理,构造成一个标准的进行了预处理的数据集。
        super(AdultDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
