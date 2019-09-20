import tensorflow
import numpy
import time

data_points = 10000
data_dim = 6

s = tensorflow.Session()

# Some random data
x = numpy.random.rand(data_points, data_dim)
y = numpy.empty((data_points, 1))

for t in range(data_points):
    y[t] = numpy.random.randint(0, 2)

# Create the placeholder and variable (weights)
x_in = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, data_dim])
y_in = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1])
W = tensorflow.Variable(tensorflow.random_normal(shape=[data_dim, 1]))
b = tensorflow.Variable(tensorflow.random_normal(shape=[1, 1]))

# The predicted y value y=1/(1+exp(-wx-b))
numerator = tensorflow.constant(1.)
denominator = tensorflow.add(1., tensorflow.exp(
    tensorflow.negative(tensorflow.add(tensorflow.matmul(x_in, W), b))))
y_predict = tensorflow.truediv(numerator, denominator)

# Cross entropy loss: yln⁡p+(1-y)ln⁡(1-p)
M = tensorflow.multiply(y_in, tensorflow.math.log(y_predict))
N = tensorflow.multiply(tensorflow.subtract(1., y_in),
                        tensorflow.math.log(tensorflow.subtract(1., y_in)))
loss_func = tensorflow.reduce_mean(tensorflow.add(M,N))

# Define opt first, then the initializer
opt = tensorflow.train.AdamOptimizer(1e-3)

train = opt.minimize(loss_func)

init = tensorflow.global_variables_initializer()
s.run(init)

for ep in range(100):
    batch_idx = numpy.random.choice(len(x), size=64)
    batch_x = numpy.array(x[batch_idx])
    batch_y = numpy.array(y[batch_idx])

    s.run(train, feed_dict={x_in: batch_x, y_in: batch_y})
    print("W", s.run(W))
    print("b", s.run(b))
