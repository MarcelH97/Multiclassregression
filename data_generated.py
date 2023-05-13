from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

from algorithms import gd,ad_grad
from loss_functions import CrossEntropy,crossentropy,crossentropygradient,softmaxvector,my_f


n_features=5
n_classes=3
n_samples=100



x1,y1 = make_classification(n_samples = n_samples,
                                       n_features = n_features,
                                       n_informative = 5,
                                       n_redundant = 0,
                                       n_classes = n_classes,
                                       weights = [.2, .3, .8])

data=x1
Features=[]
Classes=[]
for i in range(n_features):
  Features.append('Feature_'+str(i))
for i in range(n_classes):
    Classes.append('Class_'+str(i))



Data=pd.DataFrame(data, columns = Features)
Data['b'] = 1
Data['Class']=y1
Data = pd.get_dummies(Data, columns = ['Class'])
Features.append('b')
x = Data[Features].to_numpy()
y = Data[Classes].to_numpy()

n, d = x.shape

w= np.zeros(d*n_classes)
w = w.reshape(d,n_classes)


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


etst=37734