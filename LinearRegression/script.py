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
