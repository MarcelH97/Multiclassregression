import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy.linalg as la



from algorithms import gd,ad_grad
from loss_functions import CrossEntropy,crossentropy,crossentropygradient,softmaxvector,my_f


data = pd.read_csv("./datasets/iris.data", sep=",", names = ["Petal Length", "Petal Width", "Sepal Length", "Sepal width" , "Class"])
data['b'] = 1
number_of_classes=3
data = pd.get_dummies(data, columns = ['Class'])


features = ["Petal Length", "Petal Width", "Sepal Length", "Sepal width","b" ]
pred=["Class_Iris-setosa","Class_Iris-versicolor","Class_Iris-virginica"]
x1 = data[features]
y1 = data[pred]
x1.to_csv('x.csv',index=False)
y1.to_csv('y.csv',index=False)
x = pd.read_csv('x.csv')
x = x.values
y=pd.read_csv('y.csv')
y=y.values

it_max = 100
n, d = x.shape


L=1
w= np.zeros(d*number_of_classes)
w = w.reshape(d,number_of_classes)


def Loss(w):
    return CrossEntropy(x,w,y)
def gradient(w):
    return np.vstack(crossentropygradient(x,w,y))

result_gd=gd(Loss,gradient,w,numb_iter=10000)
w_gd=result_gd[2]
result_adgd=ad_grad(Loss,gradient,w,numb_iter=10000)
w_adgd=result_adgd[2]



pred_gd=softmaxvector(x,w_gd)
pred_adgd=softmaxvector(x,w_adgd)

ettet=28828