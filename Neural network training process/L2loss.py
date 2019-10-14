
# coding: utf-8

# ## Import the library

# In[69]:


import pandas as pd
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt



# ## layer definition (Need to do!!!)

# In[70]:


def InnerProduct_ForProp(x,W,b):
    m=x.shape[0]
    y = np.dot(x,W)
    for i in range(m):
        y[i,:]+b
    return y

def InnerProduct_BackProp(dEdy,x,W,b):
    dEdx = np.dot(dEdy,W.T)
    dEdW = np.dot(x.T,dEdy) 
    dEdb = np.dot(np.ones([1,48000]),dEdy)
    return dEdx,dEdW,dEdb

def L2Loss_BackProp(y,t):
    dEdx = 2*(y-t)
    return dEdx

def Sigmoid_ForProp(x):
    y=1/(1+np.exp(-x))
    return y

def Sigmoid_BackProp(dEdy,x):
    dEdx = dEdy*(1-x)*x
    return dEdx

def ReLu_ForProp(x):
    y=np.maximum(0,x)
    return y

def ReLu_BackProp(dEdy,x):
    x[x<=0]=0
    x[x>0]=1   
    dEdx = dEdy*x       
    return dEdx

def loss_ForProp(y,y_pred):
    loss = np.square(y_pred-y).sum()
    return loss


# ## Setup the Parameters and Variables (Can tune that!!!)

# In[71]:


eta =  0.000001       #learning rate
Data_num = 784      #size of input data   (inputlayer)
W1_num =  15        #size of first neural (1st hidden layer)
Out_num =   10      #size of output data  (output layer)
iteration =   1500       #epoch for training   (iteration)
image_num = 60000   #input images
test_num  = 10000   #testing images

## Cross Validation ##
##spilt the training data to 80% train and 20% valid##
train_num = int(image_num*0.8)
valid_num = int(image_num*0.2)


# ## Setup the Data (Create weight array here!!!)

# In[72]:


w_1= (np.random.normal(0,1,Data_num*W1_num)).reshape(Data_num,W1_num)/100
w_out  = (np.random.normal(0,1,W1_num*Out_num)).reshape(W1_num, Out_num)/100
b_1, b_out = randn(1,W1_num)/100,randn(1,Out_num)/100


# ## Prepare all the data

# ### Load the training data and labels from files

# In[73]:


df = pd.read_csv('fashion-mnist_train_data.csv')
fmnist_train_images = df.as_matrix()

df = pd.read_csv('fashion-mnist_test_data.csv')
fmnist_test_images = df.as_matrix()

df = pd.read_csv('fashion-mnist_train_label.csv')
fmnist_train_label = df.as_matrix()


# ### Show the 100 testing images

# In[74]:

'''
plt.figure(figsize=(20,20))
for index in range(100):
    image = fmnist_test_images[index].reshape(28,28)
    plt.subplot(10,10,index+1,)
    plt.imshow(image)
plt.show() '''


# ### Convert the training labels data type to one hot type

# In[75]:


label_temp = np.zeros((image_num,10), dtype = np.float32)
for i in range(image_num):
    label_temp[i][fmnist_train_label[i][0]] = 1
train_labels_onehot = np.copy(label_temp)


# ### Separate train_images, train_labels into training and validating 

# In[76]:


train_data_img = np.copy(fmnist_train_images[:train_num,:])
train_data_lab = np.copy(train_labels_onehot[:train_num,:])
valid_data_img = np.copy(fmnist_train_images[train_num:,:])
valid_data_lab = np.copy(train_labels_onehot[train_num:,:])
# Normalize the input data between (0,1)
train_data_img = train_data_img/255.
valid_data_img = valid_data_img/255.
test_data_img = fmnist_test_images/255.



# ## Execute the Iteration (Need to do!!!)

# In[77]:

valid_accuracy = []
train_accuracy = []
for i in range(iteration):

    # Forward-propagation
    Inner1 = InnerProduct_ForProp(train_data_img,w_1,b_1)
    Relu = ReLu_ForProp(Inner1)
    Inner2 = InnerProduct_ForProp(Relu,w_out,b_out)
    
    loss = loss_ForProp(Inner2,train_data_lab)

    # Bakcward-propagation
    gpred = L2Loss_BackProp(Inner2,train_data_lab)
    dEdx,Grad_w_out,Grad_b_out = InnerProduct_BackProp(gpred,Relu,w_out,b_out)
    ReLuback = ReLu_BackProp(dEdx,Relu)
    dEda,Grad_w_1,Grad_b_1 = InnerProduct_BackProp(ReLuback,train_data_img,w_1,b_1)
  
    # Parameters Updating (Gradient descent)
    w_1 = w_1 - eta*Grad_w_1
    b_1 = b_1 - eta*Grad_b_1
    w_out = w_out - eta*Grad_w_out
    b_out = b_out - eta*Grad_b_out    

    # Do cross-validation to evaluate model
    validInner1 = InnerProduct_ForProp(valid_data_img,w_1,b_1)
    validSig = ReLu_ForProp(validInner1)
    validInner2 = InnerProduct_ForProp(validSig,w_out,b_out)

    # Get 1-D Prediction array   
    # Compare the Prediction and validation
    loss = loss_ForProp(validInner2,valid_data_lab)
    
    
    vpred_index = np.argmax(validInner2, axis=1)
    vacc = (vpred_index == np.argmax(valid_data_lab, axis=1)).sum()   
    valid_accuracy.append(vacc/120)

    tpred_index = np.argmax(Inner2,axis=1)
    tacc = (tpred_index == np.argmax(train_data_lab,axis=1)).sum()
    train_accuracy.append(tacc/480)
    print('iteration: %d loss：%f  tacc: %f validacc: %f' %(i,loss,tacc/480,vacc/120))

testInner1 = InnerProduct_ForProp(test_data_img,w_1,b_1)
testRelu = ReLu_ForProp(testInner1)
test_Out_data = InnerProduct_ForProp(testRelu,w_out,b_out)

test_Prediction  = np.argmax(test_Out_data, axis=1)[:,np.newaxis].reshape(test_num,1)
df = pd.DataFrame(test_Prediction,columns=["Prediction"])
df.to_csv("L2loss_prediction_ID.csv",index=True, index_label="index")
    
    
accuracy = np.array(valid_accuracy)
plt.plot(accuracy, label="$iter-accuracy$")
y_ticks = np.linspace(0, 100, 11)
plt.legend(loc='best')
plt.xlabel('iteration')
plt.axis([0, iteration, 0, 100])
plt.ylabel('accuracy')
plt.show()
    

