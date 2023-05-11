import numpy as np
from loss_functions import crossentropy,crossentropygradient,softmaxvector,CrossEntropy
from algorithms import gd,ad_grad

x = np.array([[3, 2, 1, 1], [3, 3, 1, 1]])
w1=np.array([1,1,2])
w2=np.array([1,1,2])
w3=np.array([1,2,3])
b=np.array([1,1,1])
w=np.array([w1,w2,w3,b])
wtest=w.astype(float)
b=[1,1,1]
y=np.array([[1,0,0],[0,1,0]])
(m, n) = x.shape


prediction1=softmaxvector(x,w)
loss1=crossentropy(x,w,y)
loss1_1=CrossEntropy(x,w,y)



def Loss(w):
    return CrossEntropy(x,w,y)
def gradient(w):
    return np.vstack(crossentropygradient(x,w,y))





lossarray=[]

def fit(x,w,y, epochs, learn_rate):
    for epoch in range(epochs):


        grad_for_w,grad_for_b=crossentropygradient(x,w,y)

        w_update=w[0:-1]-learn_rate*grad_for_w
        b_update=w[-1]-learn_rate*grad_for_b
        w=np.append(w_update,[b_update],axis=0)

        d2=softmaxvector(x,w)
        result2=CrossEntropy(x,w,y)

        lossarray.append(result2)

    return w,lossarray

w2,loss = fit(x,w, y,epochs=100, learn_rate=0.50)

predictionfinal=softmaxvector(x,w)


result_gd=gd(Loss,gradient,wtest,numb_iter=10)
w_gd=result_gd[2]
result_adgd=ad_grad(Loss,gradient,wtest,numb_iter=10)
w_adgd=result_adgd[2]

pred_gd=softmaxvector(x,w_gd)
pred_adgd=softmaxvector(x,w_adgd)


hsfhfsh=84848
