import tensorflow as tf
import numpy as np
from generate_data import generate_training_data

tf.set_random_seed(0)

input_samples = 112
num_labels = 4
base_freq = 12

#How many samples will be used per training session
batch_size = 100

#How many samples will be used per validation session
validation_size = 1000

#Sampled noisey cosine waves will go here sampled at input_samples rate
X = tf.placeholder(tf.float32, [None, input_samples])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, num_labels])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout 
pkeep = tf.placeholder(tf.float32)

L = 28
M = 20
N = 15
O = 10
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([input_samples, L], stddev=0.1))  
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.001))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.001))
B3 = tf.Variable(tf.ones([N])/10)
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.001))
B4 = tf.Variable(tf.ones([O])/10)
W5 = tf.Variable(tf.truncated_normal([O, num_labels], stddev=0.001))
B5 = tf.Variable(tf.zeros([num_labels]))

# The model

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# initialize the graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i, base_freq, input_samples, batch_size):
    batch_X, batch_Y = generate_training_data(base_freq, batch_size, input_samples)
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0 
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i/decay_speed)
    sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep: 0.75, lr: learning_rate})


def validation_step(i, base_freq, input_samples, batch_size):
    # learning rate decay
    batch_X, batch_Y = generate_training_data(base_freq, batch_size, input_samples)
    max_learning_rate = 0.005
    min_learning_rate = 0.0001
    decay_speed = 1000
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i/decay_speed)

    print("\n Validating Model at step: "+str(i)+"\n")
    print(accuracy.eval(feed_dict={X: batch_X, Y_: batch_Y, pkeep:1.0}, session=sess))
    sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep: 0.75, lr: learning_rate})
    print(accuracy.eval(feed_dict={X: batch_X, Y_: batch_Y, pkeep:1.0}, session=sess))

for i in range(10000+1): 
    training_step(i, base_freq, input_samples, batch_size)
    if i % 100 == 0:
        validation_step(i, base_freq, input_samples, validation_size)

print("Training Complete!")
    



