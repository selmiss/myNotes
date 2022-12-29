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



