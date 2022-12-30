# Tensorflow安装和使用

## Tensorflow安装

### 使用pycharm进行安装

python版本

```
Python3.9 /opt/homebrew/bin/python3
```

tensorflow包

```
tensorflow-macos 2.11.0
```

## 一、线性回归

$$y = WX + b + \epsilon$$

> 梯度下降

$loss = \sum_i(w*x_i+b-y_i)^2$

$w'=w-lr*\frac{\partial loss}{\partial w}$

$b'=b-lr*\frac{\partial loss}{\partial b}$

$w'*x+b' -> y$

## 二、分类问题

> one-hot 编码

dog = [1,0,0...]

cat = [0,1,0...]

> Classification

$out = XW+b$

对于分类问题(单层）:

X:[b',像素点个数]

W:[像素点个数,类别个数]

b:[类别个数,1]

out:[b,类别个数]

使用激活函数relu，变为非线性

$out = relu(XW+b)$

> 欧式距离计算损失函数

loss = MSE(out,label)

手写数字分类源码实例：

```python
import tensorflow as tf
import keras.datasets as datasets
import keras.layers as layers
import keras.optimizers as optimizers
import keras

(x, y),(x_val, y_val) =datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)


model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(10)])
optimizer = optimizers.SGD(learning_rate=0.001)

def train_epoch(epoch):
    #4.循环：epoch 对一个数据集迭代依次，step对一个batch迭代一次
    for step, (x,y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28*28))
            # 1. compute output [b,784] => [b,10]
            out = model(x)
            # 2.compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        #3. optimize and update w1,w2,w3,b1,b2,b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr*grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 100 ==0:
            print(epoch, step, loss.numpy())

def train():
    for epoch in range(30):
        train_epoch(epoch)

if __name__ == '__main__':
    train()
```

## 三、数据结构

### 3.1 Data Container

- list

  [1,1.2,'hello',(1,2)]

- np.array

  不太适应GPU计算等

- **tf.Tensor**

  - scalar: 1,1
  - vector: [1,1],[1,1,2.2,...]
  - matrix:[[1.1,1.2],[3.3,4.4],[5.5, 6.6]]
  - tensor: rank > 2

### 3.2 数据类型

- int,float,double
- bool
- string

### 3.3 Tansor Property

> .device gpu() cpu()

当前数据所在硬件

```python
with tf.device("cpu")
	a=tf.constant([1])
with tf.device('gpu')
	b=tf.range(4)
a.device
b.device
aa=a.gpu() #转换为gpu
bb=b.cpu()
b.numpy() #转换为numpy类型
```

> bool is_tensor(object); 判断是否是Tensor类型

> tf.convert_to_tensor(object); 类型转换
>
> tf.convert_to_tensor(object, dtype=tf.int32);

> tf.cast(object, dtype=tf.float32); 类型转换函数

> tf.Variable

可优化参数，如W，b

```python
b=tf.Variable(a, name='input_data')
b.name #'input_data:0'
b.trainable #True
```

> Tensor to numpy

```python
a.numpy()
int(a)
float(a)
```

> 从Numpy和list生成

生成矩阵，生成0矩阵，生成数组

```python
tf.convert_to_tensor(np.ones([2,3]))
<tf.Tensor: shape=(2, 3), dtype=float64, numpy=
array([[1., 1., 1.],
       [1., 1., 1.]])>

tf.convert_to_tensor(np.zeros([2,3]))
<tf.Tensor: shape=(2, 3), dtype=float64, numpy=
array([[0., 0., 0.],
       [0., 0., 0.]])>

tf.convert_to_tensor([1,2])
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>
```

注意转换类型，以最高类型进行转换

> tf.zeros()

```python
tf.zeros([])
<tf.Tensor: shape=(), dtype=float32, numpy=0.0>

tf.zeros([1])
<tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>

tf.zeros([2,2])
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[0., 0.],
       [0., 0.]], dtype=float32)>

tf.zeros([2,3,3])
<tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],
       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]], dtype=float32)>
```

> 接受Tensor，根据传的shape生成新的数据类型
>
> tf.zeros_like(object)
>
> tf.zeros(object.shape)

```python
a=tf.zeros([2,3,3])
tf.zeros_like(a)

<tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],
       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]], dtype=float32)>
```

> tf.ones 初始化全1，用法类似

> tf.fill(shape, int) 初始化全部为指定值

> #### 随机初始化

- 正太分布

```python
tf.random.normal([2,2],mean=1, stddev=1)
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 0.19184369,  1.4655133 ],
       [-0.22810328,  0.6256025 ]], dtype=float32)>
```

mean为均值，stddev为方差，可以不指定

使用truncated_normal,截断初始化，截取正态分布开头和结尾

```python
tf.random.truncated_normal([2,2],mean=0, stddev=1)
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[-0.09060024,  0.27958587],
       [ 0.58651525,  0.7342018 ]], dtype=float32)>
```

- 均匀分布

```python
tf.random.uniform([2,2],minval=0, maxval=100)
```

### 3.4 随机打散

### 3.5 Scalar/Loss

```python
out = tf.random.uniform([4,10])
y = tf.range(4)
y = tf.one_hot(y, depth=10)
loss = tf.keras.losses.mse(y, out)
loss = tf.reduce_mean(loss)
```

### 3.6 Vector 向量

```python
net=layers.Dense(10)
net.build((4,8))
net.kernel
net.bias
<tf.Variable 'bias:0' shape=(10,) dtype=float32, numpy=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)>
```

### 3.7 Matrix

> Dim=2

> Dim=3

## 四、索引和切片

### 4.1 索引

- 基本索引:类似数组 a\[0][0]

- Numpy-style索引:a[0,0,1]

- start:end

  A[a:b] 切片，从a到b，左闭右开

  A[:b] 起始b

  A[-1:] 从右开始第一个到最后

​	A[:] 当前维度全部都取

​	A[:,:,:,0] 少了一个维度

- start: end :step

  加了取样步长，step为负则为逆序采样

- ...

  省略号，省略多个：，任意长的:

#### 4.2 Selective Indexing

- tf.gather

  取数据a的第0维的2，3个数据

  ```python
  tf.gather(a, axis=0, indices=[2,3])
  ```

- tf.gather_nd

  类似a\[0][1], a[0,1]

  ```python
  tf.gather_nd(a, [0,1])
  tf.gather_nd(a, [[0,1],[1,1]])
  #若a为2维，则返回 TensorShape([1,2])
  ```

- tf.boolean_mask

  选择在某一维是否需要哪个参数

  axis表示取哪个维度

  ```python
  tf.boolean_mask(a, mask=[True,True,False],axis=0)
  ```

## 五、Tensorflow维度变换

### 5.1 View

如手写分类图片，[b,28,28] => [b,28\*28] => [b,2,14\*28]

保证原有数据被利用，不修改原有数据，变换理解方式

```python
a=tf.random.normal([4,28,28,3])
a.shape,a.ndim
(TensorShape([4, 28, 28, 3]), 4)

tf.reshape(a,[4,784,3]).shape
TensorShape([4, 784, 3])

tf.reshape(a, [4,-1,3]).shape
TensorShape([4, 784, 3])

tf.reshape(a, [4,784*3]).shape
TensorShape([4, 2352])

tf.reshape(a, [4,-1]).shape
TensorShape([4, 2352])
```

ndim显示维度，**也可以进行反向恢复**

### 5.2 转置

维度交换,perm参数默认为倒序

```python
a=tf.random.normal((4,3,2,1))
a.shape
TensorShape([4, 3, 2, 1])

tf.transpose(a,perm=[0,1,3,2]).shape
TensorShape([4, 3, 1, 2])
```

### 5.3 增加减少维度

#### 增加维度：

```python
a=tf.random.normal([4,35,8])
tf.expand_dims(a,axis=0).shape
TensorShape([1, 4, 35, 8])

tf.expand_dims(a,axis=-1).shape
TensorShape([4, 35, 8, 1])
```

正数index在前面添加维度，负数index在后面添加维度

#### 减小维度：

只能减少shape=1的维度，才可以去掉

```python
a=tf.zeros([1,2,1,3])
tf.squeeze(a,axis=0)
<tf.Tensor: shape=(2, 1, 3),
...

tf.squeeze(a,axis=0).shape
TensorShape([2, 1, 3])

tf.squeeze(a,axis=2).shape
TensorShape([1, 2, 3])
```

不设置axis则减少所有shape为1的维度

## 六、Broadcasting

维度对齐➕扩张

[4,16,16,32] + [32] 则会对[32]扩张为[1,1,1,32] => [4,16,16,32]

然后再进行运算。

没有真正复制数据，逻辑上扩张复制。

注：右边小维度对齐才可以broadcast，或者相同维度有一个是1

> Object = tf.broadcast_to(Object, [shape])

#### tile

用法基本相同，但是实际复制了数据，没有做内存优化

## 七、数学运算

- +-*/

- **,pow,square

  平分，n次方，开方

- sqrt

- //,%

- exp,log

  ```
  tf.math.log()
  #log自然对数为底
  tf.exp()
  #e的n次方
  ```

- @,matmul

  矩阵相乘，多维数据只使用后两个维度相乘

- linear layer

## 八、前向传播-张量

使用手写数字mnist数据集进行训练

1. 引入包和数据集

   ```python
   import tensorflow as tf
   import keras.datasets as datasets
   
   (x, y), _ = datasets.mnist.load_data()
   ```

2. 对数据进行处理，转换格式，设置每步迭代长度batch

   ```python
   x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
   y = tf.convert_to_tensor(y, dtype=tf.int32)
   
   print(x.shape, y.shape, x.dtype, y.dtype)
   # 查看最小值，最大值
   print(tf.reduce_min(x), tf.reduce_max(x))
   print(tf.reduce_min(y), tf.reduce_max(y))
   
   train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
   # 使用迭代器，插看每步数据集
   train_iter = iter(train_db)
   sample = next(train_iter)
   print('batch:', sample[0].shape, sample[1].shape)
   ```

3. 初始化权重参数矩阵

   ```python
   # [b, 784] => [b, 512] => [b, 128] => [b,10]
   # 注意设定方差，否则可能不好收敛
   w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
   b1 = tf.Variable(tf.zeros([256]))
   w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
   b2 = tf.Variable(tf.zeros([128]))
   w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
   b3 = tf.Variable(tf.zeros([10]))
   lr = 1e-3
   ```

4. 前向传播迭代

   ```python
   for epoch in range(3):
       # 对数据集迭代3次
       for step, (x, y) in enumerate(train_db):
           # x :[128, 28, 28], y:[128]
           x = tf.reshape(x, [-1, 28*28])
   
           # 梯度计算
           with tf.GradientTape() as tape:
               # x :[128, 28*28], y:[128]
               h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
               h1 = tf.nn.relu(h1)
   
               h2 = h1@w2 + b2
               h2 = tf.nn.relu(h2)
               out = h2@w3 + b3
   
               # 计算loss
               y_onehot = tf.one_hot(y, depth=10)
   
               # mse = mean((y-out)^2) 求均值
               loss = tf.square(y_onehot - out)
               # mean: scalar
               loss = tf.reduce_mean(loss)
           grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
           # 更新参数,原地更新，不改变数据类型
           w1.assign_sub(lr * grads[0])
           b1.assign_sub(lr * grads[1])
           w2.assign_sub(lr * grads[2])
           b2.assign_sub(lr * grads[3])
           w3.assign_sub(lr * grads[4])
           b3.assign_sub(lr * grads[5])
   
           if step % 100 == 0:
               print(epoch, step, 'loss:', float(loss))
   ```

## 九、合并与分割

- tf.concat

  拼接操作

  ```python
  a=tf.ones([4,35,8])
  b=tf.ones([2,35,8])
  c=tf.concat([a,b], axis=0)
  c.shape
  TensorShape([6, 35, 8])
  ```

  axis指定维度

- tf.split

  按比例打散

  ```python
  a=tf.ones([3,35,4])
  res=tf.split(a, axis=1, num_or_size_splits=[10,10,15])
  len(res)
  3
  res[0].shape,res[1].shape,res[2].shape
  (TensorShape([3, 10, 4]), TensorShape([3, 10, 4]), TensorShape([3, 15, 4]))
  ```

- tf.stack

  相同的shape，创建一个新的维度,axis指定新维度的位置

  ```python
  a=tf.ones([4,35,8])
  b=tf.ones([4,35,8])
  tf.stack([a,b], axis=3).shape
  TensorShape([4, 35, 8, 2])
  ```

- tf.unstack

​	还原之前stack操作，或者按照某一个维度全部打散。

## 十、数据统计

#### 10.1 基本数据统计

- tf.norm

- reduce_min/max/mean(a, axis=0)

  最小，最大，均值，可以指定维度

- aromas/argmin

  最值所在位置，可指定维度

- tf.equal

  判断相等

- tf.unique

  不重复的值，并给出map类似的索引

  ```python
  a=tf.constant([4,2,2,4,3])
  tf.unique(a)
  Unique(y=<tf.Tensor: shape=(3,), dtype=int32, numpy=array([4, 2, 3], dtype=int32)>, idx=<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 1, 0, 2], dtype=int32)>)
  ```

#### 10.2 张量（Tensor）排序

