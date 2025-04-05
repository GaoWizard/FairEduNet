from abc import abstractmethod
from collections.abc import Hashable
from functools import wraps

from aif360.datasets import Dataset
from aif360.decorating_metaclass import ApplyDecorator


def _make_key(args, kwargs, unhashable, kwd_mark=(object(),)):
    """functools 的简化版本.给指定的函数参数生成唯一的键，键用于缓存函数的结果.
    """
    key = args
    if kwargs:
        key += kwd_mark
        for item in kwargs.items():
            if not isinstance(item[1], Hashable):
                return unhashable
            key += item
    return key

def memoize(func):
    """基于functools.lru_cache(Python 2 中不可用).
        效率较低，但在此只用于存储浮点数.
    """
    sentinal = object()
    unhashable = object()
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = _make_key(args, kwargs, unhashable)
        if key is unhashable:
            return func(*args, **kwargs)
        result = cache.get(key, sentinal)
        if result is not sentinal:
            return result
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper


BaseClass = ApplyDecorator(memoize)

class Metric(BaseClass):
    """指标的基础类."""
    @abstractmethod
    def __init__(self, dataset):
        """初始化一个 `Metrics` 抽象基础类.

        参数:
            dataset (Dataset): 用于评估参数的数据集.
        """
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError("dataset must be of Dataset class")
