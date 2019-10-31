import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

#set up hyperParameters
hidden = 100 
momentum = 0.9
learningRate = 0.1

#grab test/train D.S.
numbers = np.loadtxt('mnist_train.csv', delimiter=',')
test = np.loadtxt('mnist_test.csv', delimiter=',')

#pull targets from D.S.
labels = numbers[:,0]
targets = np.zeros((60000, 10), dtype=float)
for row in range(0, 60000):
    for col in range(0, 10):
        if labels[row] == col:
            targets[row][col] = .9
        else:
            targets[row][col] = .1
            
testTargets = np.zeros((10000, 10), dtype=float)
testLabels = test[:,0]
for row in range(0, 10000):
    for col in range(0, 10):
        if labels[row] == col:
            testTargets[row][col] = .9
        else:
            testTargets[row][col] = .1
#print("targets:\n", targets[:5])

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

#create 2-d array of random weights from [-.5-.5] for
#the inputs to the hidden layer (based on hyperParameter set above)
hiddenWeight = np.random.uniform(low=-(1/np.sqrt(hidden)), high=(1/np.sqrt(hidden)), size=(785, hidden))
outputWeight = np.random.uniform(low=-(1/np.sqrt(10)), high=(1/np.sqrt(10)), size=(hidden, 10))

#hiddenWeight = np.random.random_sample((785, hidden)) - 0.5
#outputWeight = np.random.random_sample((hidden, 10)) - 0.5

#create 2-d array of random weights from [-.5-.5] for
#the hidden neurons to the ouput (hidden x outputs)
#!!!need to look at function to finish this
#outputWeight = np.random.uniform(low=-(1/np.sqrt(10)), high=(1/np.sqrt(10)), size=(10, hidden))

#create arrays of zeros to eventually store previous
#runs weight deltas
hiddenDelta = np.zeros((hidden, 785))
outputDelta = np.zeros((hidden, 10))

for epoch in range(0, 50):
    print("epoch: ", epoch)
    correct = np.zeros(50)
    for row in range(0,60000):
        # print("inputs shape: ", inputs.shape)
        # print("hiddenWeight shape: ", hiddenWeight.shape)
        # dot product observation with set of weights for each hidden neuron
        inputvector = inputs[row]
        hiddenVector = np.dot(inputs[row], hiddenWeight)

        # print("hiddenVector shape: ", hiddenVector.shape)
        
        # run sigmoid function on hidden neuron vector
        hiddenVector = sigmoid(hiddenVector)
        
        # print("outputWeight shape: ", outputWeight.shape)

        # dot product hidden neurons with set of weights for each output neuron
        outputVector = np.dot(hiddenVector, outputWeight)    

        # print("outputVector shape: ", outputVector.shape)
        
        # run sigmoid function on hidden neuron vector
        outputVector = sigmoid(outputVector)
        
        # calculate error on output vector
        # value * (1 - value) * (target - value)
        outputError = np.add(outputVector, -1)  # not sure if this is the correct syntax (it's not)
        # print("outputVector: ", outputVector)
        # print("outputError: ", outputError)
        outputError = np.multiply(outputError, outputVector)
        subError = np.subtract(targets[row], outputVector)
        outputError = np.multiply(outputError, subError)

        #calculate error on hidden vector
        #hidden Neuron value * (1 - H.N.V) * dot(hidden neuron weights, 
        #                                   its corresponding output error) 
        hiddenError = np.add(hiddenVector, -1)
        hiddenError = np.multiply(hiddenError, hiddenVector)
        dotWeightAndOutput = np.dot(outputWeight, outputError) #what the heck is this doing
        hiddenError = np.multiply(hiddenError, dotWeightAndOutput)


        #find weight update (hidden-output) delta and store in array for momentum
        #1. calculate momentum
    #    print("outputDelta shape: ", outputDelta.shape)
        mom = np.multiply(outputDelta, momentum) 
    #    print("outputDelta shape: ", outputDelta.shape)
        outputDelta.fill(learningRate)
    #    print("outputDelta shape: ", outputDelta.shape)
    #    print("outErr.shape: ", outputError.shape)
        outputDelta = np.multiply(outputDelta, outputError)
    #    print("outputDelta shape: ", outputDelta.shape)
        #outputDelta = np.transpose(outputDelta)
        #print("outputDelta shape: ", outputDelta.shape)
        outputDelta = np.transpose(np.multiply(np.transpose(outputDelta), hiddenVector))
    #    print("outputDelta shape: ", outputDelta.shape)
        outputDelta = np.add(outputDelta, mom)
    #    print("outputDelta shape: ", outputDelta.shape)
    #    print("outputDelta head: ", outputDelta[:5])
        #update each weight by adding weight with weight delta element-wise
        outputWeight = np.add(outputWeight, outputDelta)

        #find weight update (input-hidden) delta and store in array for momentum
        #1. calculate momentum
        mom = np.multiply(hiddenDelta, momentum)
        hiddenDelta.fill(learningRate)
    #    print("hiddenDelta shape (learningRate): ", hiddenDelta.shape)
        hiddenDelta = np.multiply(np.transpose(hiddenDelta), hiddenError)
    #    print("hiddenDelta shape (mult delta and error)^trans in function: ", hiddenDelta.shape)
    #    print("hiddenError shape: ", hiddenError.shape)
        hiddenDelta = np.multiply(np.transpose(hiddenDelta), inputs[row])
    #    print("hiddenDelta shape (after mult with inputs[row])^trans in function: ", hiddenDelta.shape)
    #    print("inputs[row] shape: ", inputs[row].shape)
        hiddenDelta = np.add(hiddenDelta, mom)
    #    print("hiddenDelta shape (after add with mom): ", hiddenDelta.shape)
    #    print("mom shape: ", mom.shape)
        hiddenWeight = np.add(hiddenWeight, np.transpose(hiddenDelta))
    #    print("hiddenDelta shape (add with hidWeight)^trans in function: ", hiddenDelta.shape)

    train = np.dot(inputs, hiddenWeight)
    train = sigmoid(train)
    train = np.dot(train, outputWeight)
    train = sigmoid(train)

    for row in range(0, 60000):
        if np.argmax(targets[row]) == np.argmax(train[row]):
            correct[epoch] += 1
    print("percent correct for epoch[", epoch, "]: ", correct[epoch]/60000)

train = np.dot(inputs, hiddenWeight)
train = sigmoid(train)
train = np.dot(train, outputWeight)
train = sigmoid(train)

conMatrix = np.zeros((10, 10), dtype=int)
correct = 0
for row in range(0, 60000):
    conMatrix[np.argmax(targets[row])][np.argmax(train[row])] += 1
    if np.argmax(targets[row]) == np.argmax(train[row]):
        correct += 1
    
print("percent correct: ", correct/60000)
print("Confusion Matrix\n", conMatrix)


