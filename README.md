# ANN-BY-BACK-PROPAGATION-ALGORITHM

## Aim:
To implement multi layer artificial neural network using back propagation algorithm.


## Equipments Required:
Hardware – PCs

Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab


## Related Theory Concept:

### Weight initialization:
Set all weights and node thresholds to small random numbers. Note that the node threshold is the negative of the weight from the bias unit(whose activation level is fixed at 1).

### Calculation of Activation:
* The activation level of an input is determined by the instance presented to the network.

* The activation level oj of a hidden and output unit is determined.

### Weight training:
* Start at the output units and work backward to the hidden layer recursively and adjust weights.

* The weight change is completed.

* The error gradient is given by:

* For the output units.

* For the hidden units.

* Repeat iterations until convergence in term of the selected error criterion. An iteration includes presenting an instance, calculating activation and modifying weights.


## Algorithm
1. Import packages

2.Defining Sigmoid Function for output

3.Derivative of Sigmoid Function

4.Initialize variables for training iterations and learning rate

5.Defining weight and biases for hidden and output layer

6.Updating Weights


## Program:
```

/*
Program to implement ANN by back propagation algorithm.
Developed by   : S. SANJU
RegisterNumber :  212219040137
*/
import numpy as np
X=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
X=X/np.amax(X,axis=0)
y=y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)

epoch=7000
lr=0.1
inputlayer_neuron=2
hiddenlayer_neuron=3
output_neuron=1

wh=np.random.uniform(size=(inputlayer_neuron,hiddenlayer_neuron))
bh=np.random.uniform(size=(1,hiddenlayer_neuron))
wout=np.random.uniform(size=(hiddenlayer_neuron,output_neuron))
bout=np.random.uniform(size=(1,output_neuron))

for i in range(epoch):
    hinp1=np.dot(X,wh)
hinp=hinp1+bh
hlayer_act=sigmoid(hinp)
outinp1=np.dot(hlayer_act,wout)
outinp=outinp1+bout
output=sigmoid(outinp)

EO=y-output
outgrad=derivatives_sigmoid(output)
d_output=EO* outgrad
EH=d_output.dot(wout.T)
hiddengrad=derivatives_sigmoid(hlayer_act)
d_hiddenlayer=EH*hiddengrad
wout+=hlayer_act.T.dot(d_output)*lr
wh+=X.T.dot(d_hiddenlayer)*lr
print("Input: \n"+str(X))
print("Actual Output: \n"+str(y))
print("Predicted Output: \n",output)

```


## Output:
```
[[0.66666667 1.        ]
 [0.33333333 0.55555556]
 [1.         0.66666667]]
Actual Output: 
[[0.92]
 [0.86]
 [0.89]]
Predicted Output: 
 [[0.78079535]
 [0.76990501]
 [0.78370086]]
```


## Result:
Thus the python program successully implemented multi layer artificial neural network using back propagation algorithm.
