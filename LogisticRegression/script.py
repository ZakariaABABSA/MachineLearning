import math

w1 = 2
w2 =-1
w3 = 5
b = 1

def predict(x1,x2,x3):
    y = b + w1*x1 + w2*x2 + w3*x3
    return y

def segmoid(z):
    f = 1/(1+math.exp(-z))
    return f

z = predict(0,10,2)
probability = segmoid(z)
print("the probability is : ",probability)

