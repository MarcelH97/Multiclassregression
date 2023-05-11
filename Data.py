import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.preprocessing import OneHotEncoder



from main import Gd, Adgd, Bb, Nesterov,Armijo
from loss_functions import crossentropy,crossentropygradient,softmaxvector,my_f


data = pd.read_csv("./datasets/iris.data", sep=",", header=None, names = ["Petal Length", "Petal Width", "Sepal Length", "Sepal width" , "Class"])
data['b'] = 1
number_of_classes=3
data = pd.get_dummies(data, columns = ['Class'])


features = ["Petal Length", "Petal Width", "Sepal Length", "Sepal width","b" ]
pred=["Class_Iris-setosa","Class_Iris-versicolor","Class_Iris-virginica"]
x1 = data[features]
y1 = data[pred]
x1.to_csv('x.csv', header=False)
y1.to_csv('y.csv',header=False,index=False)
x = pd.read_csv('x.csv')
x = x.values
y=pd.read_csv('y.csv')
y=y.values

it_max = 10
n, d = x.shape


L=1
w= np.zeros(d*number_of_classes)
w = w.reshape(d,number_of_classes)

def loss_func(w):
    return crossentropy(x,w,y)


def grad_func(w):
    return crossentropygradient(x,w,y)


(m, n) = x.shape


prediction1=softmaxvector(x,w)
loss1=crossentropy(x,w,y)

grad_for_w,grad_for_b=crossentropygradient(x,w,y)

w_update=w[0:-1]-0.0001*grad_for_w
b_update=w[-1]-0.0001*grad_for_b
w_new=np.append(w_update,[b_update],axis=0)


prediction2=softmaxvector(x,w_new)
loss2=crossentropy(x,w_new,y)

w0=w

adgd = Adgd(loss_func=loss_func, grad_func=grad_func, eps=0, it_max=it_max)
adgd.run(w0=w0)



ettet=28828