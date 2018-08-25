### Tensorflow 的一般流程
- 导入/生成样本数据集
- 转换和归一化数据
- 划分样本数据集为训练样本集、测试样本集和验证样本集
- 设置机器学习参数(超参数)
- 初始化变量和占位符
- 定义模型结构
- 声明损失函数
- 初始化模型和训练模型
- 评估机器学习模型
- 调优超参数
- 发布/预测结果

### 声明张量
- 固定张量 
    + 创建制定维度的零张量 `zero_tsr = tf.zeros([row_dim,col_dim])`
    ```
    tf.zeros(
        shape,
        dtype=tf.float32,
        name=None
    )  

    tf.zeros([3, 4], tf.int32)  # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ```
    + 创建指定维度的常数填充的张量 `ones_tsr = tf.ones([row_dim,col_dim])`
    ```
    tf.ones(
        shape,
        dtype=tf.float32,
        name=None
    )  

    tf.ones([2, 3], tf.int32)  # [[1, 1, 1], [1, 1, 1]]
    ```
    + 创建指定维度的常数填充张量 `filled_tsr = tf.fill([row_dim,rol_dim])`
    ```
    tf.fill(
        dims,
        value,
        name=None
    )
    fill([2, 3], 9) ==> [[9, 9, 9], [9, 9, 9]]
    ```
    + 用已知常数张量创建一个张量 `constants_tsr = tf.constant([1,2,3])`
    ```
    tf.constant(
        value,
        dtype=None,
        shape=None,
        name='Const',
        verify_shape=False
    )
    # Constant 1-D Tensor populated with value list.
    tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

    # Constant 2-D tensor populated with value list.
    tensor = tf.constant([1, 2, 3, 4, 5, 6], shape=[2,3]) => [[1, 2, 3], [4, 5, 6]]

    # Constant 2-D tensor populated with scalar value -1. Like tf.fill
    tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                                 [-1. -1. -1.]]
    ```
- 相似形状的张量
    + 创建与给定张量形状相同的张量 
        * `zeros_similar = tf.zeros_like(constants_tsr)`
        ```
        tf.zeros_like(
            tensor,
            dtype=None,
            name=None,
            optimize=True
        )
        tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
        tf.zeros_like(tensor)  # [[0, 0, 0], [0, 0, 0]]
        ```
        * `ones_similar = tf.ones_like(constants_tsr)`
        ```
        tf.ones_like(
            tensor,
            dtype=None,
            name=None,
            optimize=True
        )
        tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
        tf.ones_like(tensor)  # [[1, 1, 1], [1, 1, 1]]
        ```
    + tips: 因为这些张量依赖给定的张量，所以初始化时需要按序进行，如果打算一次性初始化所有张量，那么程序会报错
- 序列张量
    + 创建指定间隔的张量1 `linear_tsr = tf.linspace(start=0, stop=1, start=3)`
    ```
    tf.lin_space(
        start,
        stop,
        num,
        name=None
    )
    tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
    tf.lin_space(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
   
    # 返回 start<=   <= stop 的值
    ```
    + 创建指定间隔的张量2 `integer_tsr = tf.range(start=6, limit=15, detla=3)`
    ```
     tf.range(
        start,
        limit,
        delta,
        dtype,
        name=None
    )
    start = 3
    limit = 1
    delta = -0.5
    tf.range(start, limit, delta)  # [3, 2.5, 2, 1.5]

    limit = 5
    tf.range(limit)  # [0, 1, 2, 3, 4]

    # 返回 start<=    < limit 的值
    ```
- 随机张量
    + 均匀分布的随机数 `randunif_tsr = tf.random_uniform([row_dim,col_dim], minval=0.0, maxval=1)`
    ```
    tf.random_uniform(
        shape,
        minval=0,
        maxval=None,
        dtype=tf.float32,
        seed=None,
        name=None
    )

    # 返回 minval<=   <maxval
    ```
    + 正态分布的随机数 `randnorm_tsr = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)`
    ```
    tf.random_normal(
        shape,
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32,
        seed=None,
        name=None
    )
    ```
    + 带有指定边界的正态分布随机数 `randnorm_tsr = tf.truncated_normal([row_dim, col_dim], mean=0.0, stddev=1.0)`
    ```
    tf.truncated_normal(
        shape,
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32,
        seed=None,
        name=None
    )

    # that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
    # 正态分布的随机数位于指定均值到两个标准差之间的区间
    ```
    + 张量/数组的随机化 `shuffled_output = tf.random_shuffle(input_tensor)`
    ```
    tf.random_shuffle(
        value,
        seed=None,
        name=None
    )
    [[1, 2],       [[5, 6],
     [3, 4],  ==>   [1, 2],
     [5, 6]]        [3, 4]]
    ```
    + 张量的随机裁剪 `cropped_output = tf.random_crop(my_img, [height\2, width\2, 3)`
    ```
    tf.random_crop(
        value,
        size,
        seed=None,
        name=None
    )
    ```
- 转换为张量 `tf.convert_to_tensor(arg, dtype=tf.float32)`
```
    tf.convert_to_tensor(
        value,
        dtype=None,
        name=None,
        preferred_dtype=None
    )
    
    import numpy as np
    def my_func(arg):
      arg = tf.convert_to_tensor(arg, dtype=tf.float32)
      return tf.matmul(arg, arg) + arg
    # The following calls are equivalent.
    value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
    value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
    value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    # value 可以是张量 常量数组 numpy数组
```

### 封装张量为变量
- 创建好张量之后就可以通过`tf.Variable()`函数封装张量为变量

### 占位符和变量
- 占位符
    + Tensorflow对象，用于输入输出数据的格式，允许传入指定类型和形状的数据，并依赖计算图计算结果
    + `tf.placeholder`创建占位符
    + 占位符仅仅声明数据的位置，用于传输局到计算图，占位符通过会话的feed_dict参数获取数据
    + 在计算图中使用占位符时，必须在其上执行至少一个操作
    ```
    sess=tf.Session()
    x=tf.placeholder(tf.float32, shape=[2,2])
    y=tf.identity(x)
    x_vals=np.random.rand(2,2)
    sess.run(y,feed_dict={x:x_vals})
    ```
- 变量
    + Tensorflow机器学习算法的参数，tensorflow维护(调整)这些变量的状态来优化机器学习算法
    + `tf.Variable`创建变量，过程是输入一个张量，返回一个变量
    + 创建后需要初始化变量
    ```
    my_var=tf.Variable(tf.zeros([3,2]))
    sess=tf.Session()
    initialize_op=tf.global_variables_initializer()
    sess.run(initialize_op)
    sess.run(my_var)
    ```
    + 初始化
        * 最常用的是helper函数`global_variable_initializer()`，会一次性初始化创建的所有变量
        * 给予已经初始化的变量进行初始化，必须按序初始化
        ```
        sess.tf.Session()
        first_var = tf.Variables(tf.zeros([2,3]))
        sess.run(first_var.initializer)
        second_var = tf.Variabls(tf.zeros_like(first_var))
        sess.run(second_var.initializer)
        ```
