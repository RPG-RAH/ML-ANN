#ML program
#Create a Single Perceptorn that can learns
#1.AND-Logic 2.OR-Logic 3.NAND-Logic 4.NOR-Logic

import random

#Perceptron Class that uses Gradient Descent
class Perceptron :
	def __init__(self,input_size) :
		self.input_size = input_size
		self.weights    = [random.random() for i in range(input_size)]
		self.bias       = random.random()

	def linear_combination(self,x) :
		return sum([(wi*xi) for wi,xi in zip(self.weights,x)]+[self.bias])

	def threshold(self,output) :
		return 1 if(output>=0.5) else 0		# 0.5=threshold

	def update_weights(self,eta,training_instances,train_output) :
		dw = [0 for i in range(self.input_size)]
		dbias=0
		for x,t in zip(training_instances,train_output) :
			o  = self.threshold(self.linear_combination(x))
			dw = [ wi+(eta*(t-o)*xi) for wi,xi in zip(dw,x) ]
			dbias= dbias+(eta*(t-o))
			self.weights = [ wi+dwi for wi,dwi in zip(self.weights,dw) ]
			self.bias   += dbias



#--------------------------------------------------------------------------------
#Implement Gate Logic
#1.Choose Any One Set
'''
#AND Gate
train_dataset = [(0,0),(0,1),(1,0),(1,1)]
train_output  = [0,0,0,1]
'''
#------------------------------------------------
'''
#OR Gate
train_dataset = [(0,1),(0,1),(1,0),(1,1)]
train_output  = [0,1,1,1]
'''
#------------------------------------------------
'''
#NAND
train_dataset = [(0,0),(0,1),(1,0),(1,1)]
train_output  = [1,1,1,0]
'''
#------------------------------------------------
'''
#NOR
train_dataset = [(0,0),(0,1),(1,0),(1,1)]
train_output  = [1,0,0,0]
'''

#2.After Choosing Training Set Uncomment below for training
#Main Program
'''
p = Perceptron(2)
for i in range(10000):
	p.update_weights(0.05,train_dataset,train_output)

print('w=',p.weights)
print('bias=',p.bias)
'''




#--------------------------------------------------------------------------------
#Testing with Weights we get after training
'''
#AND-Test Weights
w = [0.5829120982552319, 0.4293152424734711]
bias= -0.21466102630144362
p1 = Perceptron(2)
p1.weights=w
p1.bias=bias
print(p1.threshold(p1.linear_combination( (1,0) )))
'''
#------------------------------------------------
'''
#OR-Test Weights
w= [0.6621767106644769, 0.5614571041039128]
bias= 0.0491784100097252
p1 = Perceptron(2)
p1.weights=w
p1.bias=bias
print(p1.threshold(p1.linear_combination( (1,1) )))
'''
#------------------------------------------------
'''
#NAND-Test Weights
w= [-0.13250542524068137, -0.02005196670653496]
bias= 0.6365292561687708
p1 = Perceptron(2)
p1.weights=w
p1.bias=bias
print(p1.threshold(p1.linear_combination( (0,1) )))
'''
#------------------------------------------------

#NOR-Test Weights
w= [-0.15337854609242085, -0.26555350065013106]
bias= 0.5466025555642907
p1 = Perceptron(2)
p1.weights=w
p1.bias=bias
print(p1.threshold(p1.linear_combination( (0,0) )))		# x1=0 , x2=0	-> output=1
print(p1.threshold(p1.linear_combination( (0,1) )))		# x1=0 , x2=1	-> output=0
print(p1.threshold(p1.linear_combination( (1,0) )))		# x1=1 , x2=0	-> output=0
print(p1.threshold(p1.linear_combination( (1,1) )))		# x1=1 , x2=1	-> output=0
