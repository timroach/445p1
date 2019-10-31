import numpy as np
import sys
import idx2numpy


def sigmoid(x):
    return 1/(1+np.exp(-x))


class NeuralNetwork:

    def __init__(self, hiddenLayers, momentum, learningRate):
    
        # set up hyperParameters
        self.hidden = hiddenLayers
        self.momentum = momentum
        self.learningRate = learningRate
        self.numOutputs = 10
        
        trainingImagesFile = '../MNIST/train-images-idx3-ubyte'
        trainingLabelsFile = '../MNIST/train-labels-idx1-ubyte'
        testingImagesFile = '../MNIST/t10k-images-idx3-ubyte'
        testingLabelsFile = '../MNIST/t10k-labels-idx1-ubyte'

        # Set up training set
        self.trainingImagesRaw = idx2numpy.convert_from_file(trainingImagesFile)
        # Create target (label) list
        self.trainingLabels = idx2numpy.convert_from_file(trainingLabelsFile)
        # Convert from (1xm) to (mx1) array
        self.trainingLabels = np.reshape(self.trainingLabels, (-1, 1))
        
        # Read testing set from file
        self.testingImagesRaw = idx2numpy.convert_from_file(testingImagesFile)
        # Create target (label) list
        self.testingLabels = idx2numpy.convert_from_file(testingLabelsFile)
        # Convert from (1xm) to (mx1) array
        # self.testingLabels = np.reshape(self.trainingLabels, (-1, 1))

        self.numTrainingImages = self.trainingImagesRaw.shape[0]
        self.numTrainingInputs = self.trainingImagesRaw.shape[1] * self.trainingImagesRaw.shape[2]
        
        # Array of training targets - for each label, target array has a row
        # where row[label] = .9, all others = .1
        self.trainingTargets = np.ndarray((self.trainingLabels.shape[0], self.numOutputs))
        np.ndarray.fill(self.trainingTargets, 0.1)
        for index, row in enumerate(self.trainingTargets):
            label = self.trainingLabels[index]
            row[label] = 0.9

        # Array of testing targets - for each label, target array has a row
        # where row[label] = .9, all others = .1
        self.testingTargets = np.ndarray((self.testingLabels.shape[0], self.numOutputs))
        np.ndarray.fill(self.testingTargets, 0.1)
        for index, row in enumerate(self.testingTargets):
            label = self.testingLabels[index]
            row[label] = 0.9

        # Build training input array from 28 x 28 images
        self.trainingInputs = []
        for rawdigit in self.trainingImagesRaw.astype(float):
            test = rawdigit.flatten()
            self.trainingInputs.append(test)
        # Convert input array to numpy array
        self.trainingInputs = np.asarray(self.trainingInputs)

        # Build testing input array from 28 x 28 images
        self.testInputs = []
        for rawdigit in self.testingImagesRaw.astype(float):
            test = rawdigit.flatten()
            self.testInputs.append(test)
        # Convert input array to numpy array
        self.testInputs = np.asarray(self.testInputs)
        
        # normalize the inputs
        self.trainingInputs = np.divide(self.trainingInputs[:][:], 255)
        self.testInputs = np.divide(self.testInputs[:][:], 255)
        
        # add bias column to trainingInputs and testInputs
        bias = np.ones((60000,1))
        self.trainingInputs = np.append(self.trainingInputs, bias, 1)
        testBias = np.ones((10000,1))
        self.testInputs = np.append(self.testInputs, testBias, 1)
        
        # create two  2-d arrays of random weights for connections from 
        # the inputs to  the hidden layer. Weight ranges are chosen to 
        # ensure that the average input to the hidden layer will
        # sum to 1
        self.hiddenWeights = np.random.uniform(low=-(1/np.sqrt(self.hidden)), high=(1/np.sqrt(self.hidden)), size=(785, self.hidden + 1))
        # Same as above, but for the connections from the hidden layer 
        # to the output layer
        self.outputWeights = np.random.uniform(low=-(1/np.sqrt(10)), high=(1/np.sqrt(10)), size=(self.hidden + 1, 10))
              
        # create arrays of zeros to eventually store previous
        # runs weight deltas
        self.hiddenDelta = np.zeros((self.hidden + 1, 785))
        self.outputDelta = np.zeros((self.hidden + 1, 10))

    def train(self, iterations):
        for epoch in range(0, iterations+1):
            print("epoch: ", epoch)
            correct = np.zeros(50)
            for row, _ in enumerate(self.trainingInputs):
                # dot product observation with set of weights
                # for each hidden neuron
                hiddenVector = np.dot(self.trainingInputs[row], self.hiddenWeights)

                # run sigmoid function on hidden neuron vector
                hiddenVector = sigmoid(hiddenVector)
                hiddenVector[self.hidden] = 1

                # dot product hidden neurons with set of weights
                # for each output neuron
                outputVector = np.dot(hiddenVector, self.outputWeights)

                # Run sigmoid function on hidden neuron vector
                outputVector = sigmoid(outputVector)

                # calculate error on output vector
                outputError = np.subtract(1.0, outputVector)
                outputError = np.multiply(outputError, outputVector)
                subError = np.subtract(self.trainingTargets[row], outputVector)
                outputError = np.multiply(outputError, subError)

                # Calculate error on hidden vector:
                #   hidden Neuron value * (1 - H.N.V) *
                #   dot(hidden neuron weights, its corresponding output error)
                hiddenError = np.subtract(1, hiddenVector)
                hiddenError = np.multiply(hiddenError, hiddenVector)
                dotWeightAndOutput = np.dot(self.outputWeights, outputError)
                hiddenError = np.multiply(hiddenError, dotWeightAndOutput)

                # Find weight update (hidden-output) delta and store in array for momentum
                #   1. calculate momentum
                mom = np.multiply(self.outputDelta, self.momentum)
                self.outputDelta.fill(self.learningRate)

                #   2. Multiply previous run's output delta by momentum
                # (alpha * previous run's output delta) this is all 0s first run
                self.outputDelta = np.multiply(self.outputDelta, outputError)
                self.outputDelta = np.transpose(np.multiply(np.transpose(self.outputDelta), hiddenVector))
                self.outputDelta = np.add(self.outputDelta, mom)
                #   3. Update each weight by summing weight with
                #      weight delta element-wise
                self.outputWeights = np.add(self.outputWeights, self.outputDelta)

                # Find weight update (input-hidden) delta and store in array for momentum
                #   1. calculate momentum
                mom = np.multiply(self.hiddenDelta, self.momentum)
                self.hiddenDelta.fill(self.learningRate)
                self.hiddenDelta = np.multiply(np.transpose(self.hiddenDelta), hiddenError)
                self.hiddenDelta = np.multiply(np.transpose(self.hiddenDelta), self.trainingInputs[row])
                self.hiddenDelta = np.add(self.hiddenDelta, mom)
                self.hiddenWeights = np.add(self.hiddenWeights, np.transpose(self.hiddenDelta))

            train = np.dot(self.trainingInputs, self.hiddenWeights)
            train = sigmoid(train)
            train = np.dot(train, self.outputWeights)
            train = sigmoid(train)
    
            for row, _ in enumerate(self.trainingInputs):
                if np.argmax(self.trainingTargets[row]) == np.argmax(train[row]):
                    correct[epoch] += 1
            print("percent correct for epoch[", epoch, "]: ", correct[epoch]/60000)
    
        train = np.dot(self.trainingInputs, self.hiddenWeights)
        train = sigmoid(train)
        train = np.dot(train, self.outputWeights)
        train = sigmoid(train)

        conMatrix = np.zeros((10, 10), dtype=int)
        correct = 0
        for row in range(0, 60000):
            conMatrix[np.argmax(self.trainingTargets[row])][np.argmax(train[row])] += 1
            if np.argmax(self.trainingTargets[row]) == np.argmax(train[row]):
                correct += 1

        print("percent correct: ", correct/60000)
        print("Confusion Matrix\n", conMatrix)


def main():

    # Initialize new Multi Layer Perceptron
    # with 100 hidden layers, momentum of 0.9,
    # and a learning rate of 0.1
    mlp = NeuralNetwork(100, 0.9, 0.1)
    # Train for 10 epochs and print report
    # including confusion matrix and accuracy
    mlp.train(10)
    sys.exit("Done")


if __name__ == "__main__":
    main()
