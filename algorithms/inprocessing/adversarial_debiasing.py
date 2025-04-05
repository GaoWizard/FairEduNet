import numpy as np

# 异常处理
try:
    import tensorflow.compat.v1 as tf
except ImportError as error:
    from logging import warning
    warning("{}: 无法导入Tensorflow,AdversarialDebiasing 方法可能不可用."
            "若要安装，请运行:\n"
            "pip install 'aif360[AdversarialDebiasing]'".format(error))

from aif360.algorithms import Transformer


class FairEduNet(Transformer):
    """
    对抗性学习(Adversarial debiasing)是一种处理中方法。他通过学习一个分类器，
    能最大限度地提高预测准确率，同时降低对抗者从预测值中缺点受保护属性的能力[1].
    这个方法可以得到一个公平的分类器，因为预测值中不包含任何可被对抗者利用的群体性歧视.

    参考文献:
        [1] B. H. Zhang, B. Lemoine, and M. Mitchell, "Mitigating Unwanted
           Biases with Adversarial Learning," AAAI/ACM Conference on Artificial
           Intelligence, Ethics, and Society, 2018.
    """

# 初始化函数
    def __init__(self,
                 unprivileged_groups,
                 privileged_groups,
                 scope_name,
                 sess,
                 seed=None,
                 adversary_loss_weight=0.1,
                 num_epochs=50,
                 batch_size=128,
                 classifier_num_hidden_units=200,
                 adversarial_regularization=0.1,
                 debias=True):
        """
        参数:
            unprivileged_groups (tuple，元组形式): 非特权群体的表征.
            privileged_groups (tuple): 有特权群体的表征.
            scope_name (str): tensorflow 变量作用域的名称，字符串格式.
            sess (tf.Session): tensorflow 会话.
            seed (int, 可选项): 随机种子，用于使`predict`可重复.
            adversary_loss_weight (float, 可选项): 能够选择对抗性损失强度的超参数.
            num_epochs (int, 可选项): 训练的轮数(training epochs).
            batch_size (int, 可选项): 批处理数量(Batch size).
            classifier_num_hidden_units (int, 可选项): 分类器模型中隐藏神经元(hidden units)的数量
            debias (bool, 可选项): 学习带有偏见或去除偏见的分类器.
        """
        super(AdversarialDebiasing, self).__init__(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        # 定义一个作用域名称
        self.scope_name = scope_name
        # 设定一个随机种子
        self.seed = seed

        # 定义非特权群体和有特权群体
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        # 检查是否只有一个特权值和非特权组
        if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
            raise ValueError("此方法只支持定义一个非特权/特权群体.")
        # 获取第一个非特权群体的受保护属性名称.
        self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]

        # 定义模型的参数：
        # 会话、对抗损失权重、轮数、批处理数量、分类器隐藏神经元数量、去偏见处理
        self.sess = sess
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.debias = debias

        # 预先定义的占位符，后续操作中填充具体数据
        # 定义特征的维度
        self.features_dim = None
        # 特征的初始设置
        self.features_ph = None
        # 受保护属性的初始设置
        self.protected_attributes_ph = None
        # 真实标签的初始设置
        self.true_labels_ph = None
        # 预测标签的初始设置
        self.pred_labels = None

        # 一个控制对抗性正则化项的权重
        self.adversarial_regularization = adversarial_regularization

        self.adversary_loss_weight = adversary_loss_weight

        # 为门控器和专家网络初始化权重和偏置的列表
        self.gate_weights = []
        self.expert_weights = []

    def _classifier_model(self, features_ph, features_dim, keep_prob):
        """
        分类器，用于计算结果变量(the outcome variable)的预测值.
        接收了三个参数：
            features: 输入特征.
            features_dim: 特征维度.
            keep_prob: 在 dropout 层使用的保留概率.
        """

        self.num_experts = 8
        self.expert_hidden_units = [features_dim * 2, features_dim]
        self.gate_hidden_units = [features_dim]

        # 定义L2正则化器
        l2_regularizer = tf.keras.regularizers.l2(0.01)

        # 定义门控器的网络结构
        with tf.variable_scope("classifier_model_gate"):
            gate_hidden = features_ph
            for units in self.gate_hidden_units:
                gate_hidden = tf.keras.layers.Dense(units=units,
                                                    activation='relu',
                                                    kernel_regularizer=l2_regularizer
                                                    )(gate_hidden)
                gate_hidden = tf.keras.layers.Dropout(1 - keep_prob)(gate_hidden)  # 添加Dropout
            gate_output = tf.keras.layers.Dense(units=self.num_experts,
                                                activation=tf.nn.softmax,
                                                kernel_regularizer=l2_regularizer)(gate_hidden)


        # 定义专家网络结构
        expert_outputs = []
        self.expert_weights = []  # 初始化专家权重列表
        for expert_id in range(self.num_experts):
            with tf.variable_scope(f"classifier_model_expert_{expert_id}"):
                expert_hidden = features_ph
                for units in self.expert_hidden_units:
                    expert_hidden = tf.keras.layers.Dense(units=units,
                                                          activation=tf.nn.relu,
                                                          kernel_regularizer=l2_regularizer
                                                          )(expert_hidden)
                    expert_hidden = tf.keras.layers.Dropout(1 - keep_prob)(expert_hidden)  # 添加Dropout
                expert_output = tf.keras.layers.Dense(units=1,
                                                      kernel_regularizer=l2_regularizer)(expert_hidden)
                expert_outputs.append(expert_output)


        expert_output = tf.stack(expert_outputs, axis=1)  # shape: [batch_size, num_experts, 1]
        weighted_expert_output = tf.reduce_sum(tf.expand_dims(gate_output, axis=-1) * expert_output, axis=1)

        pred_logit = weighted_expert_output
        pred_label = tf.sigmoid(pred_logit)

        # 修正后的权重收集代码，使用正确的变量作用域路径
        self.gate_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope=f"{self.scope_name}/classifier_model_gate")

        self.expert_weights = []
        for expert_id in range(self.num_experts):
            expert_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=f"{self.scope_name}/classifier_model_expert_{expert_id}")
            self.expert_weights.append(expert_weights)

        return pred_label, pred_logit




    def _adversary_model(self, pred_logits, true_labels):
        """
        计算对抗者对受保护属性的预测值.
        参数为：
        pred_logits: 预测标签的逻辑回归值.
        true_labels: 真实标签.
        """
        # 在一个名为`adversary_model`的变量作用域中定义变量
        with tf.variable_scope("adversary_model"):
            # 定义变量c和s
            # 常量变量c，初始化为1.0
            c = tf.get_variable('c', initializer=tf.constant(1.0))
            # 计算一个sigmoid函数，用于调整 `pred_logits`.
            s = tf.sigmoid((1 + tf.abs(c)) * pred_logits)

            # 定义第二层的权重和偏置，从一个3个特征的层映射到1个特征的输出层
            W2 = tf.get_variable('W2', [3, 1],
                                 initializer=tf.initializers.glorot_uniform(seed=self.seed4))
            b2 = tf.Variable(tf.zeros(shape=[1]), name='b2')

            # 计算预测受保护属性的逻辑回归值和受保护属性标签
            pred_protected_attribute_logit = tf.matmul(tf.concat([s, s * true_labels, s * (1.0 - true_labels)], axis=1), W2) + b2
            pred_protected_attribute_label = tf.sigmoid(pred_protected_attribute_logit)

        return pred_protected_attribute_label, pred_protected_attribute_logit

    def fit(self, dataset):
        """
        使用梯度下降策略计算公平性分类器的模型参数.

        参数:
            dataset (BinaryLabelDataset):包含真实标签的数据集.

        参数:
            AdversarialDebiasing: 返回自身.
        """

        # 检查执行模式，TensorFlow要在动态执行模式下运行.
        # AdversarialDebiasing要在图执行模型下运行.
        if tf.executing_eagerly():
            raise RuntimeError("AdversarialDebiasing 方法不需要再动态执行模式(eager execution mode)下运行."
                    "要修复，在调用脚本的顶部添加`tf.disable_eager_execution()`")

        # 随机种子初始化
        if self.seed is not None:
            np.random.seed(self.seed)
        ii32 = np.iinfo(np.int32)
        self.seed1, self.seed2, self.seed3, self.seed4 = np.random.randint(ii32.min, ii32.max, size=4)

        # 将数据集中的标签映射到0和1.
        temp_labels = dataset.labels.copy()
        # 有利标签数列降维，映射到1
        temp_labels[(dataset.labels == dataset.favorable_label).ravel(),0] = 1.0
        # 非有利标签数列降维，映射到0
        temp_labels[(dataset.labels == dataset.unfavorable_label).ravel(),0] = 0.0

        # 设置TensorFlow变量作用域
        with tf.variable_scope(self.scope_name):
            # 获取训练样本数量和特征维度
            num_train_samples, self.features_dim = np.shape(dataset.features)

            # 设置占位符
            # 输入特征占位符
            self.features_ph = tf.placeholder(tf.float32, shape=[None, self.features_dim])
            # 受保护属性占位符
            self.protected_attributes_ph = tf.placeholder(tf.float32, shape=[None,1])
            # 真实标签占位符
            self.true_labels_ph = tf.placeholder(tf.float32, shape=[None,1])
            # dropout操作中的保持概率占位符
            self.keep_prob = tf.placeholder(tf.float32)


            # 获取分类器的预测和分类器的损失
            self.pred_labels, pred_logits = self._classifier_model(self.features_ph, self.features_dim, self.keep_prob)
            pred_labels_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels_ph, logits=pred_logits))

            total_loss = pred_labels_loss

            # 在计算总损失时加入正则化损失
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss += tf.reduce_sum(reg_losses)

            if self.debias:
                # 获取对抗者的预测和对抗者的损失
                pred_protected_attributes_labels, pred_protected_attributes_logits = self._adversary_model(pred_logits,
                                                                                                           self.true_labels_ph)
                pred_protected_attributes_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.protected_attributes_ph,
                                                            logits=pred_protected_attributes_logits))

                # 将对抗性损失作为正则化项加入分类器损失中
                total_loss += self.adversary_loss_weight * pred_protected_attributes_loss



            # 设置优化器和学习率
            # 创建一个不可训练，初始值为0的变量`global_step`，用于记录训练过程中的迭代次数
            global_step = tf.Variable(0, trainable=False)
            # 设置初始学习率为0.001.
            starter_learning_rate = 0.01
            starter_learning_rate_db = 0.005
            # 使用指数衰减函数来逐渐减小学习率.
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
            learning_rate_db = tf.train.exponential_decay(starter_learning_rate_db, global_step, 1000, 0.96, staircase=True)

            # 创建一个用于分类器的Adam优化器实例，使用上面定义的衰减学习率
            classifier_opt = tf.train.AdamOptimizer(learning_rate)

            # 如果去偏见，对对抗模型也创建一个学习率更小的Adam优化器.
            if self.debias:
                adversary_opt = tf.train.AdamOptimizer(learning_rate_db)

            # 提取分类器模型和对抗者模型的训练变量

            # 通过检查变量名中是否包含子串 'classifier_model' , 实现从TensorFlow的全局可训练变量中筛选出属于分类器模型的变量.
            classifier_vars = [var for var in tf.trainable_variables(scope=self.scope_name) if 'classifier_model' in var.name]

            if self.debias:
                # 筛选出属于对抗者模型的变量.
                adversary_vars = [var for var in tf.trainable_variables(scope=self.scope_name) if 'adversary_model' in var.name]
                # 获取对抗者的梯度，增加分类器损失，更新分类器的参数.
                adversary_grads = {var: grad for (grad, var) in adversary_opt.compute_gradients(pred_protected_attributes_loss,
                                                                                      var_list=classifier_vars)}
            # 归一化函数
            normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

            # 创建空列表`classifier_grads`，用于存储分类器的梯度和变量对
            classifier_grads = []

            # 计算分类器的梯度
            # 计算预测标签损失对分类器变量的梯度来遍历每个变量和相应的梯度
            for (grad,var) in classifier_opt.compute_gradients(total_loss, var_list=classifier_vars):
                # 启动去偏见：
                if self.debias:
                    # 对抗者梯度进行归一化
                    unit_adversary_grad = normalize(adversary_grads[var])

                    # 减少梯度在对抗者梯度方向上的分量以及权重调整后的对抗者梯度，来降低模型学习到的偏见.

                    # 减少或消除分类器梯度中与对抗者梯度方向相同的分量
                    # 获得分类器梯度和归一化的对抗者梯度的点积，用于表示两个梯度方向相似程度的梯度
                    # 点积乘归一化的对抗者梯度，获得方向与对抗者梯度相同，大小等于点积结果的向量
                    # 从分类器梯度中减去这个向量，在对抗者梯度的方向上减小或移除分类器梯度的分量.
                    grad -= tf.multiply(tf.reduce_sum(tf.multiply(grad, unit_adversary_grad)), unit_adversary_grad)

                    # 从分类器的梯度中减去将对抗者的梯度（经过权重调整）
                    # 超参数`self.adversary_loss_weight` 用来调整对抗者梯度的影响力。
                    grad -= tf.multiply(self.adversary_loss_weight, adversary_grads[var])

                #  将调整后的梯度和变量作为元组添加到 `classifier_grads` 列表中.
                classifier_grads.append((grad, var))

            # 使用 `apply_gradients` 方法来更新分类器的参数，其中 `global_step` 可能用于学习率衰减或跟踪训练步数.
            classifier_minimizer = classifier_opt.apply_gradients(classifier_grads, global_step=global_step)

            if self.debias:
                # 更新对抗者模型参数.
                # 创建一个上下文管理器，确保更新在对抗者模型前，先应用分类器模型的梯度更新
                with tf.control_dependencies([classifier_minimizer]):
                    # 使用对抗者优化器来最小化预测受保护属性的损失，只使用了对抗者模型的变量.
                    adversary_minimizer = adversary_opt.minimize(pred_protected_attributes_loss, var_list=adversary_vars)#, global_step=global_step)

            # 初始化全局变量：模型参数、优化器参数等
            self.sess.run(tf.global_variables_initializer())

            # print("Initial gate weights:", [w.shape for w in self.sess.run(self.gate_weights)])
            # print("Initial expert weights:",
            #       [[w.shape for w in expert] for expert in self.sess.run(self.expert_weights)])

            # 初始化局部变量：准确率、临时状态等
            self.sess.run(tf.local_variables_initializer())

            # 在此处添加诊断代码
            print("诊断 - 所有可训练变量:")
            all_vars = tf.trainable_variables()
            for var in all_vars:
                print(f"变量: {var.name}")


            # 开始训练.
            # 对num_epochs中的每个epoch：
            for epoch in range(self.num_epochs):
                # 打乱训练样本的索引,确保每个epoch数据是随机的
                shuffled_ids = np.random.choice(num_train_samples, num_train_samples, replace=False)
                epo_loss = 0
                epo_bia_loss = 0
                # 循环遍历批量数据，每个批量为batch_size
                for i in range(num_train_samples//self.batch_size):
                    # 获得当前批量的索引
                    batch_ids = shuffled_ids[self.batch_size*i: self.batch_size*(i+1)]
                    # 获得当前批量的特征
                    batch_features = dataset.features[batch_ids]
                    # 获得当前批量的标签，重塑为一个二维列向量
                    # -1表示计算维度上的元素数量，1表示第二个维度（列）的数量为1
                    batch_labels = np.reshape(temp_labels[batch_ids], [-1,1])
                    # 获得当前批量的受保护属性
                    # 选取数据集中当前批量的受保护属性，重塑为二位列向量
                    batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                                 dataset.protected_attribute_names.index(self.protected_attribute_name)], [-1,1])

                    # 获得当前批量的字典，用于在TF框架内将输入数据传递给模型
                    # 将占位符的模型特征映射到当前批次的特征
                    # 将占位符的真实标签映射到当前批次的标签
                    # 将占位符的受保护属性映射到当前批次的受保护属性
                    # 控制dropout正则化中的保留概率为0.8
                    batch_feed_dict = {self.features_ph: batch_features,
                                       self.true_labels_ph: batch_labels,
                                       self.protected_attributes_ph: batch_protected_attributes,
                                       self.keep_prob: 0.8}
                    # 启用去偏见训练：
                    if self.debias:
                        # 前两个下划线用于忽略classifier_minimizer和adversary_minimizer的返回值，优化器通常不反悔有用的信息
                        # 获得预测标签损失（分类器损失）和受保护属性损失（对抗者损失）的计算结果
                        _, _, pred_labels_loss_value, pred_protected_attributes_loss_vale = self.sess.run([classifier_minimizer,
                                       adversary_minimizer,
                                       pred_labels_loss,
                                       pred_protected_attributes_loss], feed_dict=batch_feed_dict)
                        epo_loss += pred_labels_loss_value
                        epo_bia_loss += pred_protected_attributes_loss_vale

                    # 未启用去偏见训练：
                    else:
                        # 获得分类器损失的结果
                        _, pred_labels_loss_value = self.sess.run(
                            [classifier_minimizer,
                             pred_labels_loss], feed_dict=batch_feed_dict)
                        epo_loss += pred_labels_loss_value
                if self.debias:
                    print("epoch %d; iter: %d; batch classifier mean loss: %f; batch adversarial mean loss: %f" % (
                    epoch, i, epo_loss / (i + 1),
                    epo_bia_loss / (i + 1)))
                else:
                    print("epoch %d; iter: %d; batch classifier mean loss: %f" % (
                        epoch, i, epo_loss / (i + 1)))

        # 在此处添加诊断代码
        print("诊断 - 所有可训练变量:")
        all_vars = tf.trainable_variables()
        for var in all_vars:
            print(f"变量: {var.name}")


        return self

    def predict(self, dataset):
        """
        使用学习到的公平分类器，获取提供数据集的预测.

        参数:
            dataset (BinaryLabelDataset类): 包含需要转换的标签的数据集.

        返回值:
            dataset (BinaryLabelDataset类): 转换后的数据集.
        """

        # 随机种子设置
        if self.seed is not None:
            np.random.seed(self.seed)

        # 初始化

        # 获取测试样本的数量
        num_test_samples, _ = np.shape(dataset.features)

        # 初始化一个变量来跟踪已处理的样本数量
        samples_covered = 0
        # 初始化一个存储预测标签的列表
        pred_labels = []

        # 逐批处理数据，并进行预测，每批次的大小为batch_size
        while samples_covered < num_test_samples:
            # 控制每批次的开始数和结束数
            start = samples_covered
            end = samples_covered + self.batch_size
            if end > num_test_samples:
                end = num_test_samples
            # 获得当前批次的索引
            batch_ids = np.arange(start, end)
            # 获得当前批次的特征
            batch_features = dataset.features[batch_ids]
            # 获得当前批次的标签
            batch_labels = np.reshape(dataset.labels[batch_ids], [-1,1])
            # 获得当前批次的受保护属性
            batch_protected_attributes = np.reshape(dataset.protected_attributes[batch_ids][:,
                                         dataset.protected_attribute_names.index(self.protected_attribute_name)], [-1,1])

            # 获得当前批次的字典
            batch_feed_dict = {self.features_ph: batch_features,
                               self.true_labels_ph: batch_labels,
                               self.protected_attributes_ph: batch_protected_attributes,
                               self.keep_prob: 1.0}

            # [:,0]：选择预测结果数组中的第一列，并将数组转换为了一个list列表，诸葛添加到pred_labels列表中.
            pred_labels += self.sess.run(self.pred_labels, feed_dict=batch_feed_dict)[:,0].tolist()
            # 更新已处理的样本数量的变量samples_covered
            samples_covered += len(batch_ids)

        # 带有新标签的更公平的、突变的数据集

        # 获得一个新的数据集副本：是原始数据集的深拷贝
        dataset_new = dataset.copy(deepcopy = True)
        # 获得新数据集的预测分数
        dataset_new.scores = np.array(pred_labels, dtype=np.float64).reshape(-1, 1)
        # 获得新数据集的预测标签，大于0.5的预测标签为正类，小于0.5的是负类
        dataset_new.labels = (np.array(pred_labels)>0.5).astype(np.float64).reshape(-1,1)

        # 将数据集标签映射为原始值

        # 创建变量temp_labels，是新数据集labels的副本
        temp_labels = dataset_new.labels.copy()
        # temp_labels中的正类标签，降维，映射到原始数据集的有利标签
        temp_labels[(dataset_new.labels == 1.0).ravel(), 0] = dataset.favorable_label
        # temp_labels中，标签值为0的，降维，原始数据集的非有利标签
        temp_labels[(dataset_new.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        # 复制temp_labels，给新数据集的标签
        dataset_new.labels = temp_labels.copy()

        return dataset_new

    def analyze_gate_network(self):
        """分析门控网络对输入特征的重要性"""
        gate_importance = []

        if not self.gate_weights:
            print("警告：门控网络权重为空，请确保模型已训练且权重收集正确")
            return gate_importance

        # 分析每层权重矩阵
        for i, weight_tensor in enumerate(self.gate_weights):
            if 'kernel' in weight_tensor.name:  # 只分析权重矩阵，不分析偏置
                weights = self.sess.run(weight_tensor)
                # 计算每个特征的重要性（取绝对值的平均）
                feature_importance = np.mean(np.abs(weights), axis=1)
                gate_importance.append(feature_importance)

        return gate_importance

    def analyze_expert_weights(self):
        """分析每个专家网络对输入特征的重要性"""
        expert_importance = []

        if not self.expert_weights or all(not expert for expert in self.expert_weights):
            print("警告：专家网络权重为空，请确保模型已训练且权重收集正确")
            return expert_importance

        # 对每个专家
        for expert_id, expert_weight_list in enumerate(self.expert_weights):
            layer_importance = []

            # 分析每层权重矩阵
            for weight_tensor in expert_weight_list:
                if 'kernel' in weight_tensor.name:  # 只分析权重矩阵，不分析偏置
                    weights = self.sess.run(weight_tensor)
                    # 计算每个特征的重要性（取绝对值的平均）
                    feature_importance = np.mean(np.abs(weights), axis=1)
                    layer_importance.append(feature_importance)

            expert_importance.append(layer_importance)

        return expert_importance

    def get_feature_importance(self, dataset):
        """获取每个特征对模型预测的整体重要性"""
        if not self.gate_weights or all(not expert for expert in self.expert_weights):
            print("警告：模型权重为空，请确保模型已训练且权重收集正确")
            return []

        # 获取特征名称
        feature_names = dataset.feature_names

        # 获取门控网络和专家网络的重要性
        gate_importance = self.analyze_gate_network()
        expert_importance = self.analyze_expert_weights()

        # 合并所有重要性分数
        feature_importance = np.zeros(len(feature_names))

        # 添加门控网络的重要性（如果有）
        if gate_importance and len(gate_importance[0]) == len(feature_names):
            feature_importance += gate_importance[0]

        # 添加每个专家网络的第一层重要性（如果有）
        for expert_id, expert_layers in enumerate(expert_importance):
            if expert_layers and len(expert_layers[0]) == len(feature_names):
                feature_importance += expert_layers[0]

        # 归一化
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)

        return list(zip(feature_names, feature_importance))
