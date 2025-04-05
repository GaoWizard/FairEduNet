from abc import abstractmethod
from functools import wraps

from aif360.datasets import Dataset
from aif360.decorating_metaclass import ApplyDecorator

# TODO: Use sklearn.exceptions.NotFittedError?
class NotFittedError(ValueError, AttributeError):
    """
        当 `predict` 或 `transform` 在 `fit` 之前实现时报错.
    """

def addmetadata(func):
    """
    用于执行转换并返回一个新数据集的实例方法的装饰器.
    自动填充新数据集中的 `metadata` ，以反映转换后的详细信息.例如:
        {
            'transformer': 'TransformerClass.function_name',
            'params': kwargs_from_init,
            'previous': [all_datasets_used_by_func]
        }
    """
    # 使用functools.wraps decorator来保留func函数的元信息(比如名称、文档字符串等).
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        new_dataset = func(self, *args, **kwargs)
        # 检查是否是Dataset对象
        if isinstance(new_dataset, Dataset):
            new_dataset.metadata = new_dataset.metadata.copy()
            # 更新新数据集的metadata,加入执行本次转换的相关信息:transformer名称、参数、原始数据集.
            new_dataset.metadata.update({
                'transformer': '{}.{}'.format(type(self).__name__, func.__name__),
                'params': self._params,
                'previous': [a for a in args if isinstance(a, Dataset)]
            })
        return new_dataset
    return wrapper


BaseClass = ApplyDecorator(addmetadata)

class Transformer(BaseClass):
    """
    转换方法(transformers)的抽象基本类 (Abstract base class).

    转换方法(Transformers)是对 **任意作用在 :obj:`Dataset` 并返回一个新的修改的数据集** 的过程的抽象.
        这个定义涵盖了预处理、处理中和处理后算法(pre-processing, in-processing, and post-processing algorithms).
    """

    # 抽象方法注释
    @abstractmethod
    def __init__(self, **kwargs):
        """
        初始化一个转换方法的obj.

        特定算法的配置参数应当在这个方法中进行传递.
        """
        self._params = kwargs

    def fit(self, dataset):
        """
        在输入中训练一个模型.

        参数:
            dataset (Dataset): 输入数据集.

        返回值:
            Transformer: 返回转换方法，即 Transformer 本身.
        """
        return self

    def predict(self, dataset):
        """
        返回一个新的数据集,新的数据集通过在输入上运行这个转换方法得到预测标签.

        参数:
            dataset (Dataset): 输入数据集.

        返回值:
            Dataset: 输出数据集. `metadata` 应该更新以反应这次转换方法的细节.
        """
        raise NotImplementedError("'predict' is not supported for this class. "
            "Perhaps you meant 'transform' or 'fit_predict' instead.")

    def transform(self, dataset):
        """
        返回一个新的数据集,新数据集由在输入上运行这个转换方法生成.

        这个函数能返回不同的`dataset.features`,`dataset.labels`，或两者都返回.

        参数:
            dataset (Dataset): 输入数据集.

        Returns:
            Dataset: 输出数据集. `metadata` 应该更新以反应这次转换方法的细节.
        """
        raise NotImplementedError("'transform' is not supported for this class."
            " Perhaps you meant 'predict' or 'fit_transform' instead?")

    def fit_predict(self, dataset):
        """
        在输入上训练模型，并预测标签.

        等价于先调用`fit(dataset)`, 再调用 `predict(dataset)`.

        参数:
            dataset (Dataset): 输入数据集.

        返回值:
            Dataset: 输出数据集. . `metadata` 应该更新以反应这次转换方法的细节.
        """
        return self.fit(dataset).predict(dataset)

    def fit_transform(self, dataset):
        """
        在输入上训练一个模型,并相应的转换这个数据集.

        等效于先调用`fit(dataset)` ,再调用 `transform(dataset)`.

        参数:
            dataset (Dataset): 输入数据集.

        返回值:
            Dataset: 输出数据集. . `metadata` 应该更新以反应这次转换方法的细节.
        """
        return self.fit(dataset).transform(dataset)
