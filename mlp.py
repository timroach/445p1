import numpy as np
import math

def sigmoid(x):
    retrn 1/(1+np.exp(-x))

#set up hyperParameters
hidden = 20
momentum = 0.9
learningRate = 0.1

#grab test/train D.S.
numbers = np.loadtxt('mnist_train.csv', delimiter=',')
test = np.loadtxt('mnist_test.csv', delimiter=',')

#pull targets from D.S.
targets = numbers[:,0]
targets = targets.astype(int)
testTargets = test[:,0]
testTargets = testTargets.astype(int)

#pull inputs from D.S.
inputs = numbers[:,1:]
testInputs = test[:,1:]

#normalize the inputs
inputs = np.divide(inputs[:][:], 255)
testInputs = np.divide(testInputs[:][:], 255)

#add bias column to inputs
bias = np.ones((60000,1))
inputs = np.append(inputs, bias, 1)
testBias = np.ones((10000,1))
testInputs = np.append(testInputs, testBias, 1)
#print(inputs.shape)

#print(inputs[1:2])

#create 2-d array of random weights from [-.5-.5] for
#the inputs to the hidden layer (based on hyperParameter set above)
hiddenWeight = np.random.uniform(low=-(1/np.sqrt(hidden)), high=(1/np.sqrt(hidden)), size=(785, hidden))

#create 2-d array of random weights from [-.5-.5] for
#the hidden neurons to the ouput (hidden x outputs)
#!!!need to look at function to finish this
#outputWeight = np.random.uniform(low=-(1/np.sqrt(10)), high=(1/np.sqrt(10)), size=(10, hidden))

#create arrays of zeros to eventually store previous
#runs weight deltas
hiddenDelta = np.zeros((hidden, 785))
outputDelta = np.zeros((10, hidden))

for row in inputs:
    #dot product observation with set of weights for each hidden neuron
    hiddenVector = np.dot(inputs[row], hiddenWeight)
    
    #run sigmoid function on hidden neuron vector
    hiddenVector = sigmoid(hiddenVector)

    #dot product hidden neurons with set of weights for each output neuron
    outputVector = np.dot(hiddenVector, outputWeight)    

    #run sigmoid function on hidden neuron vector
    outputVector = sigmoid(outputVector)
    
    #calculate error on output vector
    #value * (1 - value) * (target - value)
    outputError = np.add(outputVector, -1)  #not sure if this is the correct syntax
    outputError = np.multiply(outputError, outputVector)
    subError = np.subtract(target[row], outputVector)
    outputError = np.multiply(outputError, subError)

    #calculate error on hidden vector
    #hidden Neuron value * (1 - H.N.V) * dot(hidden neuron weights, 
    #                                   its corresponding output error) 
    hiddenError = np.add(hiddenVector, -1)
    hiddenError = np.multiply(hiddenError, hiddenVector)
    dotWeightAndOutput = np.dot(outputWeight, outputError)
    hiddenError = np.multiply(hiddenError, dotWeightAndOutput)

    #find weight update (hidden-output) delta and store in array for momentum
    #1. calculate momentum
    mom = np.multiply(outputDelta, momentum)
    #2. calculate learning rate * vector error element-wise
    outputDelta = np.multiply(outputError, learningRate)#ok to begin modifying outputDelta
    #3. calculate delta(last operation) * hidden vector element-wise
    outputDelta = np.multiply(outputDelta, hiddenVector)
    #4. calculate delta + mom
    outputDelta = np.add(outputDelta, mom)

    #update each weight by adding weight with weight delta element-wise
    outputWeight = np.add(outputWeight, outputDelta)

    #find weight update (input-hidden) delta and store in array for momentum
    #1. calculate momentum
    mom = np.multiply(hiddenDelta, momentum)
    #2. calculate learning rate * vector error element-wise
    hiddenDelta = np.multiply(hiddenError, learningRate)#ok to begin modifying hiddenDelta
    #3. calculate delta(last operation) * input vector element-wise
    hiddenDelta = np.multiply(hiddenDelta, inputs[row])
    #4. calculate delta + mom
    hiddenDelta = np.add(hiddenDelta, mom)


    #update each weight by adding weight with weight delta element-wise
    outputWeight = np.add(outputWeight, hiddenDelta)


