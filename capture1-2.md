###  操作(计算)矩阵
- 创建矩阵
    + numpy数组
    + 创建张量的函数(`zeros()`等)
    + `diag()`函数从一个一维数组来创建对角矩阵
```
sess=tf.Session()

identity_matrix = tf.diag([1.0,1.0,1.0])
A = tf.truncated_normal([2,3])
B = tf.fill([2,3],0.8)
C = tf.random_uniform([3,2])
D = tf.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-7.,-1.],[0.,5.,-2.]]))

print(sess.run(identity_matrix))
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]

print(sess.run(A))
[[ 0.23909441  1.24854553 -0.87404215]
[-0.90024459 -0.24650373  0.97486657]]

print(sess.run(B))
[[ 0.80000001  0.80000001  0.80000001]
 [ 0.80000001  0.80000001  0.80000001]]

print(sess.run(C))
[[ 0.85576129  0.550439  ]
 [ 0.91171062  0.59328604]
 [ 0.57833397  0.23308027]]

print(sess.run(D))
[[ 1.  2.  3.]
 [-3. -7. -1.]
 [ 0.  5. -2.]]
```
- 矩阵加减法
```
print(sess.run(A+B))    # print(sess.run(tf.add(A,B)))
print(sess.run(B-B))    # print(sess.run(tf.subtract(A,B)))
```
- 矩阵乘法
```
print(sess.run(tf.matmul(B,C))) 
```
- 矩阵转置
```
print(sess.run(tf.transpose(D)))
[[ 1. -3.  0.]
 [ 2. -7.  5.]
 [ 3. -1. -2.]]
```
- 矩阵行列式
```
print(sess.run(tf.matrix_determinant(D)))
-38.0
```
- 逆矩阵
```
print(sess.run(tf.matrix_inverse(D)))
[[-0.5        -0.5        -0.5       ]
 [ 0.15789474  0.05263158  0.21052632]
 [ 0.39473684  0.13157895  0.02631579]]

# 使用cholesky矩阵分解法求逆矩阵，矩阵需要对称正定矩阵或者可以进行LU分解？？？
```
- 矩阵分解？？？
```
# cholesky矩阵分解法
print(sess.run(tf.cholesky(identity_matrix)))
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]
```
- 矩阵特征值和特征向量
```
print(sess.run(tf.self_adjoint_eig(D)))
(array([-10.65907521,  -0.22750691,   2.88658212]), 
array([[ 0.21749542,  0.63250104, -0.74339638],
       [ 0.84526515,  0.2587998 ,  0.46749277],
       [-0.4880805 ,  0.73004459,  0.47834331]]))
```
