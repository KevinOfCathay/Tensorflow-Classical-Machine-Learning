import tensorflow
from sklearn import preprocessing
import numpy

data_points = 10000
data_dim = 6

s = tensorflow.Session()

# Some random data
x = numpy.random.rand(data_points, data_dim)
y = numpy.random.rand(data_points, 1)

# Polynomial
P = preprocessing.PolynomialFeatures(3, interaction_only = True)
x = P.fit_transform(x)
dim = x.shape[1]

# Create the placeholder and variable (weights)
x_in = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, dim])
y_in = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1])
W = tensorflow.Variable(tensorflow.random_normal(shape=[dim, 1]))
b = tensorflow.Variable(tensorflow.random_normal(shape=[1, 1]))

# The predicted y value y=xW+b
y_predict = tensorflow.add(tensorflow.matmul(x_in, W), b)

# L1 loss, l = |y-y^|
loss_func = tensorflow.reduce_mean(tensorflow.abs(y_in - y_predict))

# Define opt first, then the initializer
opt = tensorflow.train.AdamOptimizer(1e-3)
train = opt.minimize(loss_func)

# Add operations into the chart
init = tensorflow.global_variables_initializer()
s.run(init)

for ep in range(100):
    batch_idx = numpy.random.choice(len(x), size=64)
    batch_x = numpy.array(x[batch_idx])
    batch_y = numpy.array(y[batch_idx])

    s.run(train, feed_dict={x_in: batch_x, y_in: batch_y})
    print("W", s.run(W))
    print("b", s.run(b))