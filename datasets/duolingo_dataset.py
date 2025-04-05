import os

import pandas as pd

from aif360.datasets import StandardDataset

default_mappings = {
    'label_maps': [{1.0: 'yes', 0.0: 'no'}],
    'protected_attribute_maps': [{0.0: 'eng', 1.0: 'noneng'}]
}

def default_preprocessing(df):
    return df

class DuolingoDataset(StandardDataset):
    """多邻国学习数据集

    文件位置:`Dataset/Duolingo LanguageLearning/DataSet and Format.md`.
    """

    def __init__(self, label_name='prediction',
                 favorable_classes=['yes'],
                 protected_attribute_names=['ui_binary','learning_binary'],
                 privileged_classes=['noneng'],
                 instance_weights_name=None,
                 categorical_features=['is_workday'],
                 features_to_keep=['is_workday', 'history_seen', 'history_correct', 'session_seen', 'delta','ui_binary','learning_binary'],
                 features_to_drop=[], na_values=[],
                 custom_preprocessing= default_preprocessing,
                 metadata=default_mappings):

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'duolingo', 'duolingo_all.csv')

        try:
            df = pd.read_csv(filepath)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://dataverse.harvard.edu/dataverse/duolingo")
            print("\n After process, place the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'duolingo'))))
            import sys
            sys.exit(1)

        super(DuolingoDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)