import numpy as np
print("----- the predicted and real values -----")
bias = 34
weight = -4.6
def predict(x1):
	y = bias + x1*weight
	return y

featurs = [3.5 , 3.69 , 3.44 , 3.43 , 4.34 , 4.42 , 2.37 ]
real_labels = [18 ,15 ,18 ,16 ,15 ,14 ,24 ]
for x, label in zip(featurs,real_labels):
	y = predict(x)
	print(" the weight :",x," | real MPG :",label , " | predicted MPG :",y)
print("----- calculating the loss ------")
actual = np.array([18, 15, 18, 16, 15, 14, 24])
predicted = np.array([
	predict(x) for x in featurs
])
diff = actual - predicted
print("L1: ")
abs_l1 = np.abs(diff)
sum_l1 = np.sum(abs_l1)
print(sum_l1)
print("MAE: ")
MAE = np.mean(abs_l1)
print(MAE)
print("L2: ")
square_l2 = diff ** 2
sum_l2 = np.sum(square_l2)
print(sum_l2)
print("MSE: ")
MSE = np.mean(square_l2)
print(MSE)
print("RMSE: ")
RMSE = np.sqrt(MSE)
print(RMSE)
