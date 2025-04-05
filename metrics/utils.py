"""这是用于执行度量指标的辅助脚本."""
import numpy as np


def compute_boolean_conditioning_vector(X, feature_names, condition=None):
    """计算布尔条件向量

    参数:
        X (numpy.ndarray): 数据集的特征
        feature_names (list): 特征的名称
        condition (list(dict)): 指定我们使用的实例的子集.格式为一个`dicts`的列表，键是`feature_names`，值是`X`中的值.
                                列表中的元素是用 OR 连接符连接的子句，每个 dict 中的键值对是用 AND 运算符连接的.
                                更多详情参见案例，如果是`None`，条件指定整个实例集`X`.

    返回值:
        numpy.ndarray(bool):布尔条件向量.形状是`[n]`，其中`n`是`X.shape[0]`.
                            如果对应行满足条件，值是`True`；反之则为 `False` .


    案例:
        >>> condition = [{'sex': 1, 'age': 1}, {'sex': 0}]
        这相当于`(sex == 1 AND age == 1) OR (sex == 0)`.
    """
    if condition is None:
        return np.ones(X.shape[0], dtype=bool)

    overall_cond = np.zeros(X.shape[0], dtype=bool)
    # group 是从 condition 列表中迭代出来的每一个元素.
    for group in condition:
        group_cond = np.ones(X.shape[0], dtype=bool)
        for name, val in group.items():
            index = feature_names.index(name)
            # 更新 group_cond 向量,使用逻辑AND操作来确保group_cond只包含那些在当前特征上值与val相等的实例.
            group_cond = np.logical_and(group_cond, X[:, index] == val)
        # 使用逻辑OR操作更新 overall_cond 向量，将满足当前 group 子条件的实例加入到结果中.
        overall_cond = np.logical_or(overall_cond, group_cond)
    return overall_cond

def compute_num_instances(X, w, feature_names, condition=None):
    """ 计算实例的数量，数学语言:`n`,可以以受保护属性为条件.
        与常规的计数不同，这里的每个实例都有一个权重，所以实际上它计算的是满足条件的实例的加权数量.

    参数:
        X (numpy.ndarray): 数据集特征.
        w (numpy.ndarray): 实例权重向量.
        feature_names (list): 特征的名称.
        condition (list(dict)): 和函数 `compute_boolean_conditioning_vector` 一样的形式.


    返回值:
        int: 实例的数量(条件可选).
    """

    # 如果有需要，可加条件可选项
    cond_vec = compute_boolean_conditioning_vector(X, feature_names, condition)
    # 计算所选权重的总和，即满足条件的实例的加权数量.
    # w[cond_vec]: 这部分使用布尔索引从权重向量w中选择满足条件的实例的权重.
    return np.sum(w[cond_vec], dtype=np.float64)

def compute_num_pos_neg(X, y, w, feature_names, label, condition=None):
    """计算正例(positives,数学语言：`P`)和负例(negatives,数学语言：`N`)的数量，可选择条件为受保护属性.

    参数:
        X (numpy.ndarray): 数据集特征.
        y (numpy.ndarray): 标签向量.
        w (numpy.ndarray): 实例权重向量.
        feature_names (list): 特征的名称.
        label (float): 标签的值(有利标签/正例 或 非有利标签/负例 ).
        condition (list(dict)): 和函数 `compute_boolean_conditioning_vector` 一样的形式.

    返回值:
        int: 正例/负例的数量(条件可选项).
    """
    y = y.ravel()
    cond_vec = compute_boolean_conditioning_vector(X, feature_names,
        condition=condition)
    return np.sum(w[np.logical_and(y == label, cond_vec)], dtype=np.float64)

def compute_num_TF_PN(X, y_true, y_pred, w, feature_names, favorable_label,
                      unfavorable_label, condition=None):
    """计算 真/假 阳性/阴性 (true/false positives/negatives)，可选择条件为受保护属性.

    参数:
        X (numpy.ndarray): 数据集特征.
        y_true (numpy.ndarray): 真(True)标签向量.
        y_pred (numpy.ndarray): 预测(Predicted) 标签向量.
        w (numpy.ndarray): 实例权重向量 - 真和预测. 数据集应当有一些实例的权重等级.
        feature_names (list): 特征的名称.
        favorable_label (float): 有利的/阳性 标签的值.
        unfavorable_label (float): 非有利的/阴性 标签的值.
        condition (list(dict)): 和函数 `compute_boolean_conditioning_vector` 一样的形式.

    返回值:
        正例/负例 的数量(条件可选项)
    """
    # 如果有需要，可加条件可选项
    cond_vec = compute_boolean_conditioning_vector(X, feature_names,
        condition=condition)

    # 防止Numpy的广播行为
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # 得到 真实阳性、真实阴性、预测阳性、预测阴性的布尔向量.
    y_true_pos = (y_true == favorable_label)
    y_true_neg = (y_true == unfavorable_label)
    # 考虑条件可选项
    y_pred_pos = np.logical_and(y_pred == favorable_label, cond_vec)
    y_pred_neg = np.logical_and(y_pred == unfavorable_label, cond_vec)

    # True/false positives/negatives
    # 真/假 阳性/阴性
    return dict(
        TP=np.sum(w[np.logical_and(y_true_pos, y_pred_pos)], dtype=np.float64),
        FP=np.sum(w[np.logical_and(y_true_neg, y_pred_pos)], dtype=np.float64),
        TN=np.sum(w[np.logical_and(y_true_neg, y_pred_neg)], dtype=np.float64),
        FN=np.sum(w[np.logical_and(y_true_pos, y_pred_neg)], dtype=np.float64)
    )

def compute_num_gen_TF_PN(X, y_true, y_score, w, feature_names, favorable_label,
                    unfavorable_label, condition=None):
    """ 计算泛用的 真/假 阳性/阴性 的数量
        可选以受保护属性为条件.泛用的计算是基于分数而非硬性预测

    参数:
        X (numpy.ndarray): 数据集特征.
        y_true (numpy.ndarray):真 标签向量.
        y_score (numpy.ndarray):预测标签向量.值的取值范围为(0,1)，
                                0 表示非有利标签的预测值. 1 表示有利标签的预测值
        w (numpy.ndarray): 实例权重向量 - 真实 和 预测
                           数据集应该有一些实例权重等级
        feature_names (list): 特征的名称.
        favorable_label (float): 有利的/阳性 的值.
        unfavorable_label (float): 非有利的/阴性 的值.
        condition (list(dict)): 和函数 `compute_boolean_conditioning_vector` 一样的形式.

    返回值:
        阳性/阴性 的数量(条件可选项).
    """
    # 如果有需要，可加条件可选项
    cond_vec = compute_boolean_conditioning_vector(X, feature_names,
        condition=condition)

    # 防止Numpy的广播行为
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    w = w.ravel()

    y_true_pos = np.logical_and(y_true == favorable_label, cond_vec)
    y_true_neg = np.logical_and(y_true == unfavorable_label, cond_vec)

    # Generalized true/false positives/negatives
    # 泛化的 真/假 阳性/阴性
    return dict(
        GTP=np.sum((w*y_score)[y_true_pos], dtype=np.float64),
        GFP=np.sum((w*y_score)[y_true_neg], dtype=np.float64),
        GTN=np.sum((w*(1.0-y_score))[y_true_neg], dtype=np.float64),
        GFN=np.sum((w*(1.0-y_score))[y_true_pos], dtype=np.float64)
    )

def compute_distance(X_orig, X_distort, X_prot, feature_names, dist_fun,
                     condition=None):
    """ 计算两组向量的元素距离.

    参数:
        X_orig (numpy.ndarray): 原始的特征.
        X_distort (numpy.ndarray): 变形后的(Distorted) 特征. 形状必须匹配 `X_orig`.
        X_prot (numpy.ndarray): 受保护属性(用于计算条件). 应和原始特征和变形后的特征一致
        feature_names (list): 受保护的特征的名称.
        dist_fun (function): 函数，返回两个一维数组之间的距离(浮点数，float).
                            (例如 :func:`scipy.spatial.distance.euclidean`).
        condition (list(dict)): 和函数 `compute_boolean_conditioning_vector` 一样的形式.

    返回值:
        (numpy.ndarray(numpy.float64), numpy.ndarray(bool)):
            * 元素距离(一维).
            * 条件向量(一维).
    """
    # 条件可选项
    cond_vec = compute_boolean_conditioning_vector(X_prot, feature_names,
        condition=condition)

    # 实例数量，即原始数据集的行数
    num_instances = X_orig[cond_vec].shape[0]
    # 初始化距离
    distance = np.zeros(num_instances, dtype=np.float64)
    for i in range(num_instances):
        # 计算每个实例的原始数据和变形后的数据的距离
        distance[i] = dist_fun(X_orig[cond_vec][i], X_distort[cond_vec][i])

    return distance, cond_vec
