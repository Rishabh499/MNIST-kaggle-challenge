import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

f=pd.read_csv("mnist/train.csv")
xt,xte,yt,yte=train_test_split(f.loc[:,f.columns!="label"],f.iloc[:,0],random_state=0)
xt=xt.as_matrix()
xt=xt.reshape(31500,28,28)

xt=xt/255
xte=xte.as_matrix()
xte=xte.reshape(10500,28,28)

xte=xte/255

yt=pd.get_dummies(yt)
yte=pd.get_dummies(yte)

def create_xy():
    x=tf.placeholder(tf.float32,[None,28,28])
    xr=tf.reshape(x,[-1,28,28,1])
    
    y=tf.placeholder(tf.float32,[None,10])
    return xr,y
def para():
    tf.set_random_seed(1)
    w1=tf.get_variable("w1",[8,8,1,8],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    w2=tf.get_variable('w2',[4,4,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    para={"w1":w1,"w2":w2}
    return para

def forward(x,para):
    w1=para['w1']
    w2=para['w2']
    z1=tf.layers.conv2d(x,padding='same',strides=1,activation=tf.nn.relu,name='z1',kernel_size=3,filters=32)
    
    p1=tf.nn.max_pool(z1,ksize=(1,2,2,1),strides=[1,2,2,1],padding='SAME')
    
    z2=tf.layers.conv2d(p1, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),padding='same',strides=1,activation=tf.nn.relu,name='z2',filters=64,kernel_size=6,use_bias=True)
    z20=tf.layers.conv2d(z2,padding='same',strides=1,activation=tf.nn.relu,name='z20',filters=128,kernel_size=8)# check this for para
    p2=tf.nn.max_pool(z20,ksize=(1,2,2,1),strides=[1,2,2,1],padding='SAME')
    p=tf.contrib.layers.flatten(p2)
    """p=tf.reshape(p2,shape=[-1,6*6*64])"""
    z1=tf.contrib.layers.fully_connected(p,120,activation_fn=None)
    z2=tf.contrib.layers.fully_connected(z1,40,activation_fn=None)
    z3=tf.contrib.layers.fully_connected(z2,10,activation_fn=None)
    return z3
def compute_cost(z3, Y):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z3, labels=Y))
    return cost
def random_mini_batches(x,y,seed,mini_batch_size):
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = x.shape[0]
    # number of training examples
    mini_batches = []
    x=np.array(x)
    y=np.array(y)
    permutation=list(np.random.permutation(m))
    s_x=x[permutation,:,:]
    s_y=y[permutation,:]
    num_complete_minibatches=int(m/mini_batch_size)
    for i in range(num_complete_minibatches):
        x_mini=s_x[i*mini_batch_size:(i+1)*mini_batch_size,:,:]
        
        x_mini.reshape(x_mini.shape[0],28,28,1)
        y_mini=s_y[i*mini_batch_size:(i+1)*mini_batch_size,:]
        mini_batch = (x_mini, y_mini)

        mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
           
            mini_batch_X = s_x[num_complete_minibatches * mini_batch_size:,:,:]
            mini_batch_Y = s_y[num_complete_minibatches * mini_batch_size:,:]
            mini_batch_X=mini_batch_X.reshape(mini_batch_X.shape[0],28,28,1)
            mini_batch = (mini_batch_X, mini_batch_Y)
            
            mini_batches.append(mini_batch)
    
    return mini_batches


def model(lr=.0001,num_epochs=500,minibatch_size=32,print_cost=True):
    x,y=create_xy()
    m=31500
    costs=[]
    parameters=para()
    z3=forward(x,parameters)
    cost=compute_cost(z3,y)
    optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    seed=2
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost=0
            num_minibatches = int(m / minibatch_size)
            np.random.seed(seed)
            
            seed+=1
            mini_batches=random_mini_batches(xt,yt,seed,32)
                
            for mini in mini_batches:
                    mini_x,mini_y=mini
                    mini_x=mini_x.reshape(mini_x.shape[0],28,28,1)
                    
                    sp,co=sess.run([optimizer,cost],feed_dict={x:mini_x,y:mini_y})
                        
                    epoch_cost+=float(float(co)/num_minibatches)
            if print_cost == True and epoch % 10 == 0:
                                print ('{},{}'.format(epoch,epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                            costs.append(epoch_cost)
model()


