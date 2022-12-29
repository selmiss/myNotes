

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





