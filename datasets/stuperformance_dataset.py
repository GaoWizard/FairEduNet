import os

import pandas as pd

from aif360.datasets import StandardDataset

default_mappings = {
    'label_maps': [{1.0: 'yes', 0.0: 'no'}],
    'protected_attribute_maps': [{0.0: 'F', 1.0: 'M'}]
}

def default_preprocessing(df):
    return df

class StuperDataset(StandardDataset):
    """葡萄牙语学生学习数据集

    原始数据集文件位置:`Dataset/student performance/readme.md`.
    """

    def __init__(self, label_name='G3_prediction',
                 favorable_classes=['yes'],
                 protected_attribute_names=['sex'],
                 privileged_classes=['1'],
                 instance_weights_name=None,
                 categorical_features=['sex'],
                 features_to_keep=[
                     'sex', 'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                     'absences', 'G1', 'G2', 'schoolsup_encoded', 'famsup_encoded', 'paid_encoded', 'activities_encoded',
                     'nursery_encoded', 'higher_encoded', 'internet_encoded', 'romantic_encoded', 'school_MS', 'address_U',
                     'famsize_LE3', 'Pstatus_T', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_mother',
                     'guardian_other', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_health', 'Fjob_other',
                     'Fjob_services', 'Fjob_teacher', 'G3_prediction'
                 ],
                 features_to_drop=[], na_values=[],
                 custom_preprocessing= default_preprocessing,
                 metadata=default_mappings):

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'stuper', 'stuper_pre.csv')

        try:
            df = pd.read_csv(filepath)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the orginal dataset and do process")
            print("\n After process, place the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'stuper'))))
            import sys
            sys.exit(1)

        super(StuperDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)