# encoding:utf-8

import tensorflow as tf
import numpy as np

BATCH_SIZE=8

rng=np.random.RandomState(23456)

X=rng.rand(32,2)
#Y=[[(X0+X1-0,05+rng.rand()*0.1)] for (X0,X1) in X ]
Y=[[X0+X1] for (X0,X1) in X ]


print "X=",X
print "Y=",Y

x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

w=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y=tf.matmul(x,w)

loss=tf.reduce_mean(tf.square(y-y_))
step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	STEPS=12000
	for i in range(STEPS):
		start=(i*BATCH_SIZE)%32
		end=start+BATCH_SIZE
		
		sess.run(step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if 0==i%500:
			total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
			print "total_loss=",total_loss

	print "w=",sess.run(w)
