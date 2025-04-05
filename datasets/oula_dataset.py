import os

import pandas as pd

from aif360.datasets import StandardDataset

default_mappings = {
}

def default_preprocessing(df):
    return df

class OULADataset(StandardDataset):
    """
    OULAD数据集
    """

    def __init__(self, label_name='final_result',
                 favorable_classes=['1'],
                 protected_attribute_names=['gender'],
                 privileged_classes=['0'],
                 instance_weights_name=None,
                 categorical_features=['gender'],
                 features_to_keep=['code_presentation','id_assessment','assessment_type',
                                   'date','weight','module_presentation_length',
                                   'id_student','date_submitted','is_banked','score','gender','region',
                                   'highest_education','imd_band','age_band','num_of_prev_attempts',
                                   'studied_credits','disability','final_result'],
                 features_to_drop=[], na_values=[],
                 custom_preprocessing= default_preprocessing,
                 metadata=default_mappings):

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'oula', 'oulad.csv')

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

        super(OULADataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)