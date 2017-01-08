
# coding: utf-8

# In[1]:

#conv Neural Network
# tensorboard --logdir=/home/ncc/notebook/learn/tensorboard/log
"""
created by kim Seong jung

"""
import numpy as np 
import tensorflow as tf

import math
import time
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os 

file_locate='/media/seongjung/Seagate Backup Plus Drive/data/test/'
sess = tf.InteractiveSession()
test_img=np.load(file_locate+'test_img.npy');
print np.shape(test_img)
img_row = np.shape(test_img)[1]
img_col = np.shape(test_img)[2]

batch_size=30
print img_row ,img_col
n_classes =2
in_ch =3
out_ch1=100
out_ch2=100
out_ch3=100
out_ch4=100
out_ch5=100


fully_ch1=1024
fully_ch2 =1024
fully_ch3 =1024



strides_1=[1,2,2,1]
strides_2=[1,1,1,1]
strides_3=[1,1,1,1]
strides_4=[1,1,1,1]
strides_5=[1,1,1,1]


x= tf.placeholder("float",shape=[None,img_col , img_row , 3],  name = 'x-input')
y_=tf.placeholder("float",shape=[None , n_classes] , name = 'y-input')
keep_prob = tf.placeholder("float")

x_image= tf.reshape(x,[-1,img_row,img_col,3])

iterate=300




weight_row =3 ; weight_col=3

pooling_row_size1=int(img_row/2)
pooling_row_size2=int(pooling_row_size1/2)
pooling_row_size3=int(pooling_row_size2/2)
pooling_row_size4=int(pooling_row_size3/2)
pooling_row_size5=int(pooling_row_size4/2)
pooling_col_size1=int(img_col/2)
pooling_col_size2=int(pooling_col_size1/2)
pooling_col_size3=int(pooling_col_size2/2)
pooling_col_size4=int(pooling_col_size3/2)
pooling_col_size5=int(pooling_col_size4/2)

print img_col , img_row


# In[2]:

with tf.device('/gpu:3'):
    #with tf.device('/gpu:1'):
    train_img=np.load(file_locate+'train_img.npy');
    train_lab=np.load(file_locate+'train_lab.npy');
    val_img= np.load(file_locate+'val_img.npy');
    val_lab = np.load(file_locate+'val_lab.npy');
    test_img=np.load(file_locate+'test_img.npy');
    test_lab=np.load(file_locate+'test_lab.npy');

    print "Training Data",np.shape(train_img)
    print "Training Data Label",np.shape(train_lab)
    print "Test Data Label",np.shape(test_lab)
    print "val Data Label" , np.shape(val_img)

    n_train= np.shape(train_img)[0]
    n_train_lab = np.shape(train_lab)[0]


# In[3]:

"""def weight_variable(name,shape):
    #initial = tf.truncated_normal(shape , stddev=0.1)
    initial = tf.get_variable(name,shape=shape , initializer = tf.contrib.layers.xavier_initializer())
    return tf.Variable(initial)"""
with tf.device('/gpu:0'):
    def bias_variable(shape):
        initial = tf.constant(0.1 , shape=shape)
        return tf.Variable(initial)



# In[4]:

with tf.device('/gpu:0'):
    def next_batch(batch_size , image , label):

        a=np.random.randint(np.shape(image)[0] -batch_size)
        batch_x = image[a:a+batch_size,:]
        batch_y= label[a:a+batch_size,:]
        return batch_x, batch_y


# In[5]:

with tf.device('/gpu:0'):

    def conv2d(x,w,strides_):
        return tf.nn.conv2d(x,w, strides = strides_, padding='SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x , ksize=[1,2,2,1] ,strides = [1,2,2,1] , padding = 'SAME')


# In[6]:

with tf.variable_scope("layer1") as scope:
    try:
        w_conv1 = tf.get_variable("W1",[weight_row,weight_col,3,out_ch1] , initializer = tf.contrib.layers.xavier_initializer())
    except:
        scope.reuse_variables()
        w_conv1 = tf.get_variable("W1",[weight_row,weight_col,3,out_ch1] , initializer = tf.contrib.layers.xavier_initializer())
with tf.variable_scope("layer1") as scope:
    try:
        b_conv1 = bias_variable([out_ch1])
    except:
        scope.reuse_variables()
        b_conv1 = bias_variable([out_ch1])
                
            
            
with tf.variable_scope('layer2') as scope:
    try:
        w_conv2 = tf.get_variable("W2",[weight_row,weight_col,out_ch1,out_ch2] , initializer = tf.contrib.layers.xavier_initializer())
    except:
        scope.reuse_variables()
        w_conv2 = tf.get_variable("W2",[weight_row,weight_col,out_ch1,out_ch2] , initializer = tf.contrib.layers.xavier_initializer())
        
with tf.variable_scope('layer2') as scope:
    try:
        b_conv2= bias_variable([out_ch2])
    except:
        scope.reuse_variables()
        b_conv2= bias_variable([out_ch2])
                
with tf.variable_scope('layer3') as scope:
    try:
        w_conv3 = tf.get_variable("W3" ,[weight_row,weight_col,out_ch2,out_ch3] , initializer = tf.contrib.layers.xavier_initializer())
    except:
        scope.reuse_variables()
        w_conv3 = tf.get_variable("W3" ,[weight_row,weight_col,out_ch2,out_ch3] , initializer = tf.contrib.layers.xavier_initializer())
with tf.variable_scope('layer3') as scope:
    try:
        b_conv3 = bias_variable([out_ch3])
    except:
        scope.reuse_variables()
        b_conv3 = bias_variable([out_ch3])
        
with tf.variable_scope('layer4') as scope:
    try:
        w_conv4 =tf.get_variable("W4" ,[weight_row,weight_col,out_ch3,out_ch4] , initializer = tf.contrib.layers.xavier_initializer())
    except:
        scope.reuse_variables()
        w_conv3 = tf.get_variable("W4" ,[weight_row,weight_col,out_ch2,out_ch3] , initializer = tf.contrib.layers.xavier_initializer())
with tf.variable_scope('layer4') as scope:
    try:
        b_conv4 = bias_variable([out_ch4])
    except:
        scope.reuse_variables()
        b_conv3 = bias_variable([out_ch3])
        
with tf.variable_scope('layer5') as scope:
    try:
        w_conv5 = tf.get_variable("W5",[weight_row,weight_col,out_ch4,out_ch5] , initializer = tf.contrib.layers.xavier_initializer())
    except:
        scope.reuse_variables()
        w_conv3 = tf.get_variable("W5" ,[weight_row,weight_col,out_ch2,out_ch3] , initializer = tf.contrib.layers.xavier_initializer())
with tf.variable_scope('layer5') as scope:
    try:
        b_conv5 = bias_variable([out_ch5])
    except:
        scope.reuse_variables()
        b_conv3 = bias_variable([out_ch3])
                


# In[7]:

#conncect hidden layer 
with tf.device('/gpu:0'):
    h_conv1 = tf.nn.relu(conv2d(x_image , w_conv1 ,strides_1)+b_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1 , w_conv2 ,strides_2)+b_conv2)
    h_conv2 = max_pool_2x2(h_conv2)#pooling
    
    h_conv3 = tf.nn.relu(conv2d(h_conv2 , w_conv3,strides_3)+b_conv3)
    h_conv4 = tf.nn.relu(conv2d(h_conv3 , w_conv4,strides_4)+b_conv4)
    h_pool4 = max_pool_2x2(h_conv4) #pooling 

    h_conv5 = tf.nn.relu(conv2d(h_conv4, w_conv5,strides_5)+b_conv5)
    h_conv5= max_pool_2x2(h_conv5) #pooling 

    print h_conv1
    print h_conv2
    print h_conv3
    print h_conv4
    print h_conv5
    
    
    end_conv = h_conv5
    #print conv2d(h_pool1 , w_conv2).get_shape()


# In[8]:

print w_conv1.get_shape()


# In[9]:

end_conv_row=int(h_conv5.get_shape()[1])
end_conv_col=int(h_conv5.get_shape()[2])
end_conv_ch=int(h_conv5.get_shape()[3])
#connect fully connected layer 
with tf.device('/gpu:0'):
    with tf.variable_scope("fc1") as scope:
        try:
            w_fc1=tf.get_variable("fc1_W",[end_conv_col*end_conv_row*end_conv_ch,fully_ch1] , initializer = tf.contrib.layers.xavier_initializer())
        except:
            scope.reuse_variables()
            w_fc1=tf.get_variable("fc1_W",[end_conv_col*end_conv_row*end_conv_ch,fully_ch1] , initializer = tf.contrib.layers.xavier_initializer())
        try:
            b_fc1 = bias_variable([fully_ch1])
        except:
            scope.reuse_variables()
            b_fc1 = bias_variable([fully_ch1])

        
with tf.device('/gpu:0'): # flat conv layer 
    end_flat_conv =tf.reshape(end_conv, [-1,end_conv_col*end_conv_row*end_conv_ch])
   
with tf.device('/gpu:0'): # connect flat layer with fully  connnected layer 
    h_fc1 = tf.nn.relu(tf.matmul(end_flat_conv , w_fc1)+ b_fc1)
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob)


# In[10]:

with tf.device('/gpu:0'):
    with tf.variable_scope('fc2') as scope:
        try:
            w_fc2 =tf.get_variable("fc2_W",[fully_ch1 , fully_ch2],initializer = tf.contrib.layers.xavier_initializer())
        except:
            scope.reuse_variables()
            w_fc2 =tf.get_variable("fc2_W",[fully_ch1 , fully_ch2],initializer = tf.contrib.layers.xavier_initializer())
        try:
            b_fc2 = bias_variable([fully_ch2])
        except:
            scope.reuse_variables()
            b_fc2 = bias_variable([fully_ch2])

with tf.device('/gpu:0'):  # join flat layer with fully  connnected layer 
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1 , w_fc2)+b_fc2)
    h_fc2= tf.nn.dropout(h_fc2 , keep_prob)
    


# In[11]:

with tf.device('/gpu:0'):
    with tf.variable_scope('fc3') as scope:
        try:
            w_fc3 =tf.get_variable("fc3_W",[fully_ch2 , fully_ch3],initializer = tf.contrib.layers.xavier_initializer())
        except:
            scope.reuse_variables()
            w_fc3 =tf.get_variable("fc3_W",[fully_ch2 , fully_ch3],initializer = tf.contrib.layers.xavier_initializer())
        try:
            b_fc3 = bias_variable([fully_ch3])
        except:
            scope.reuse_variables()
            b_fc3 = bias_variable([fully_ch3])

with tf.device('/gpu:0'):  # join flat layer with fully  connnected layer 
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2 , w_fc3)+b_fc3)
    h_fc3= tf.nn.dropout(h_fc3 , keep_prob)
    


# In[12]:

end_fc=h_fc3


# In[13]:

with tf.device('/gpu:0'):
    with tf.variable_scope('fc3') as scope:
        try:
            w_end =tf.get_variable("end_W",[fully_ch3 , n_classes ],initializer = tf.contrib.layers.xavier_initializer())
        except:
            scope.reuse_variables()
            w_end =tf.get_variable("end_W",[fully_ch3 , n_classes],initializer = tf.contrib.layers.xavier_initializer())
        try:
            b_end = bias_variable([n_classes])
        except:
            scope.reuse_variables()
            b_end = bias_variable([n_classes])

with tf.device('/gpu:0'):  # join flat layer with fully  connnected layer 
    y_conv = tf.matmul(h_fc3 , w_end)+b_end
    


# In[14]:

#dirname = '/home/ncc/notebook/mammo/result/'

dirname='/home/seongjung/바탕화면/thyroid/result'

count=0
while(True):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        break
    elif not os.path.isdir(dirname + str(count)):
        dirname=dirname+str(count)
        os.mkdir(dirname)
        break
    else:
        count+=1
print 'it is recorded at :'+str(count)


# In[15]:

f=open(dirname+"/log.txt",'w')


# In[16]:

with tf.device('/gpu:0'):
#sm_conv= tf.nn.softmax(y_conv)
    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    start_time = time.time()

    regular=0.01*(tf.reduce_sum(tf.square(y_conv)))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( y_conv, y_))
with tf.device('/gpu:0'):
    cost = cost+regular
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost) #1e-4
    with tf.name_scope("accuracy"):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv,1) ,tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction , "float")) 

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

for i in range(iterate):
    
    batch_xs , batch_ys = next_batch(batch_size, train_img , train_lab)
   # batch_val_xs  , batch_val_ys = next_batch(20 , val_img , val_lab)
    if i%100 ==0: # in here add to validation 
        try:
            val_accuracy = sess.run( accuracy , feed_dict={x:val_img , y_:val_lab , keep_prob: 1.0})        
            val_loss = sess.run(cost , feed_dict = {x:val_img , y_: val_lab , keep_prob: 1.0})
            
            train_accuracy = sess.run( accuracy , feed_dict={x:batch_xs , y_:batch_ys , keep_prob: 1.0})        
            train_loss = sess.run(cost , feed_dict = {x:batch_xs, y_: batch_ys, keep_prob: 1.0})

            #result = sess.run(sm_conv , feed_dict = {x:val_img , y_:batch_ys , keep_prob :1.0})
            print("step %d , training  accuracy %g" %(i,train_accuracy))
            print("step %d , loss : %g" %(i,train_loss))
            train_str = 'step:\t'+str(i)+'\tval_loss:\t'+str(train_loss) +'\tval accuracy:\t'+str(train_accuracy)+'\n'
          
            print("step %d , validation  accuracy %g" %(i,val_accuracy))
            print("step %d , validation loss : %g" %(i,val_loss))
            val_str = 'step:\t'+str(i)+'\tval_loss:\t'+str(val_loss) +'\tval accuracy:\t'+str(val_accuracy)+'\n'
            
            
            f.write(val_str)
            f.write(train_str)
        except :
            list_acc=[]
            list_loss=[]
            n_divide=len(val_img)/batch_size
            for j in range(n_divide):
                
                # j*batch_size :(j+1)*batch_size
                val_accuracy,val_loss = sess.run([accuracy ,cost], feed_dict={x:val_img[ j*batch_size :(j+1)*batch_size] , y_:val_lab[ j*batch_size :(j+1)*batch_size ] , keep_prob: 1.0})        
                list_acc.append(float(val_accuracy))
                list_loss.append(float(val_loss))
            val_accuracy , val_loss=sess.run([accuracy,cost] , feed_dict={x:val_img[(j+1)*batch_size : ] , y_:val_lab[(j+1)*(batch_size) : ] , keep_prob : 1.0})
            #right above code have to modify
            
            list_acc.append(val_accuracy)
            list_loss.append(val_loss)
            list_acc=np.asarray(list_acc)
            list_loss= np.asarray(list_loss)
            
            val_accuracy=np.mean(list_acc)
            val_loss = np.mean(list_loss)
            
            #result = sess.run(sm_conv , feed_dict = {x:val_img , y_:batch_ys , keep_prob :1.0})
            
            train_accuracy = sess.run( accuracy , feed_dict={x:batch_xs , y_:batch_ys , keep_prob: 1.0})        
            train_loss = sess.run(cost , feed_dict = {x:batch_xs, y_: batch_ys, keep_prob: 1.0})

            print("step %d , training  accuracy %g" %(i,train_accuracy))
            print("step %d , loss : %g" %(i,train_loss))
            train_str = 'step:\t'+str(i)+'\tval_loss:\t'+str(train_loss) +'\tval accuracy:\t'+str(train_accuracy)+'\n'
            
            print("step %d , validation  accuracy %g" %(i,val_accuracy))
            print("step %d , validation loss : %g" %(i,val_loss))
            val_str = 'step:\t'+str(i)+'\tval_loss:\t'+str(val_loss) +'\tval accuracy:\t'+str(val_accuracy)+'\n'
           
            
            f.write(val_str)
            f.write(train_str)
    
    sess.run(train_step ,feed_dict={x:batch_xs , y_:batch_ys , keep_prob : 0.7})
print("--- Training Time : %s ---" % (time.time() - start_time))
train_time="--- Training Time : ---:\t" +str(time.time() - start_time)
f.write(train_time)


# In[17]:

try:
    test_accuracy = sess.run( accuracy , feed_dict={x:test_img , y_:test_lab , keep_prob: 1.0})        
    test_loss = sess.run(cost , feed_dict = {x:test_img , y_: test_lab , keep_prob: 1.0})

    #result = sess.run(sm_conv , feed_dict = {x:test_img , y_:batch_ys , keep_prob :1.0})
    print("step %d , testidation  accuracy %g" %(i,test_accuracy))
    print("step %d , testidation loss : %g" %(i,test_loss))
    test_str = 'step:\t'+str(i)+'\ttest_loss:\t'+str(test_loss) +'\ttest accuracy:\t'+str(test_accuracy)+'\n'

    f.write(test_str)
except :
    list_acc=[]
    list_loss=[]
    n_divide=len(test_img)/batch_size
    for j in range(n_divide):

        # j*batch_size :(j+1)*batch_size
        test_accuracy,test_loss = sess.run([accuracy ,cost], feed_dict={x:test_img[ j*batch_size :(j+1)*batch_size] , y_:test_lab[ j*batch_size :(j+1)*batch_size ] , keep_prob: 1.0})        
        list_acc.append(float(test_accuracy))
        list_loss.append(float(test_loss))
    test_accuracy , test_loss=sess.run([accuracy,cost] , feed_dict={x:test_img[(j+1)*batch_size : ] , y_:test_lab[(j+1)*(batch_size) : ] , keep_prob : 1.0})
    #right above code have to modify

    list_acc.append(test_accuracy)
    list_loss.append(test_loss)
    list_acc=np.asarray(list_acc)
    list_loss= np.asarray(list_loss)

    test_accuracy=np.mean(list_acc)
    test_loss = np.mean(list_loss)

    #result = sess.run(sm_conv , feed_dict = {x:test_img , y_:batch_ys , keep_prob :1.0})
    print("step %d , testidation  accuracy %g" %(i,test_accuracy))
    print("step %d , testidation loss : %g" %(i,test_loss))
    test_str = 'step:\t'+str(i)+'\ttest_loss:\t'+str(test_loss) +'\ttest accuracy:\t'+str(test_accuracy)+'\n'

    f.write(test_str)


# In[18]:

sess.close()

