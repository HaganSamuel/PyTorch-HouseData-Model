import torch
from torch import nn # The foundation of all learning models.
from torch import optim # A package implementing various optimization algorithms.
from pathlib import Path

from os import path
import pandas as pd

import matplotlib.pyplot as plt

print(torch.version.__version__)

import graphPlotting as gP
#from graphPlotting import plottingChart as pt

# Creating the known parameters (The ones we're trying to reach from our linear regression AI model)
# https://youtu.be/V_xro1bcAuA?t=18285
weight = 0.7
bias = 0.3

'''
PyTorch model building essentials:
torch.nn - Contains all of the building blocks for computational graphs. (A neural network can be considered a computational graph)
torch.nn.Parameter - What parameters shoudl our model try and learn, often a PyTorch layer from torch.nn will set these for us.
Torch.nn.Module - The base clas for all neural network modules. If you subclass it, you should overwrite forward.
def forward() - All nn.Module subclasses require you to overwrite forward(), this method defines what happens in the forward computation

What our model does:
* Start with random values (weight & bias)
* Look at training data and adjust the random values to better represent (or get closer to) the ideal values (the weight & bias values we used to create the data)
* Learn a representation of the input (our inputted data aka HouseData.csv) and how it maps to the output (Derived from LinearDataVariant from secondDataBase).
'''

# Create train/test data split.
dataFile = pd.read_csv('LinearRegressionHouseDatabase.csv')

houseTest = torch.tensor(dataFile.to_numpy(), dtype=torch.float64)
LinearDataVariant = weight * houseTest + bias # This is the ideal output, but we won't always have access to this in the "wild" so to speak when creating the model.

print(houseTest)

# Separating the training data from the study data by 80%.
splitTrainingData = int(0.8*len(houseTest))

# We're creating a training data set (The first set of variables) to be used as the meat and potatoes of what is the correct info to be trained on, and then using 20% of the other data (Second set of variables) as the testing portion to see if it works correctly or not.
TrainingData, IdealTrainingData = houseTest[:splitTrainingData], LinearDataVariant[:splitTrainingData] # Colin ":" positioning determines the first or last of the values based on the number value fo splitTrainingData.
TestingData, IdealTestingData = houseTest[splitTrainingData:], LinearDataVariant[splitTrainingData:] # These datasets are used to test and learn upon the training data.

print(f"The current sizes of each dataset: \n TrainingData{TrainingData.type()} size: {TrainingData.size()} \n IdealTrainingData{IdealTrainingData.type()} size: {IdealTrainingData.size()} \n TestingData{TestingData.type()} size: {TestingData.size()} \n IdealTestingData{IdealTestingData.type()} size: {IdealTestingData.size()}")

# Luckily for us, PyTorch has already implemented Gradient Descent and Backpropagation for us.

# OOP (Object Oriented Programming)
class LinearRegressionModel(nn.Module): # Inherits from nn.Module (Almost everything in PyTorch inherits from nn.Module) (The lego building bricks of the PyTorch Model) (Can stack these modules to make complex neural networks)
    def __init__(self): # Required python syntax. Called when a object of the class is created (Like a default constructor). Self parameter represents the newly created object.
        super().__init__() # Calls the __init__() method of the parent class. Allows the child class to inherit the attributes and methods of the parent class.

        # Initialize model parameters for weight and bias.

        # Start with a random weight and try to adjust it to the ideal weight.
        self.weights = nn.Parameter(torch.rand(1, # nn.Parameter is a kind of tensor that is to be considered a module parameter. When assigned, they are automatically added to the list of its paramters.
                                              requires_grad=True, # If the parameter requires gradient.
                                              dtype=torch.float)) # The data type default. Can also just be "float32".

        # Start with a random bias and try to adjust it to the ideal bias.
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    # Forward method to define the computation performed at every call in the model. NEEDS TO BE DEFINED IF YOU'RE USING A Subclass "nn.Module"
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: torch.Tensor basically means x inherits as torch.Tensor by returning it. "x" is the input data.
        return self.weights * x + self.bias # The linear regression formula. You could also replace the entire line with "return self.Linear_layer(x)".
    # It should be noted that weight and bias are learnable parameters. In this model, we can decide what we want our weight and bias to be as described on line 12.

# Create a random seed for generating random numbers.
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
houseModel = LinearRegressionModel()

# Check out the parameters within the nn.Module subclass we created
list(houseModel.parameters())

# Train Model
'''
Training for a model is to move from unknown parameters to known parameters. From a poor representation of the data to a better representation of the data.

To measure the poor representation of model predictions, we use a loss function.

An optimizer takes into account the loss of a model and adjusts the model's parameters (e.g. weight & bias in our case) to improve the loss function.

More specifically, for PyTorch, we will need a training loop and a testing loop.

numpy() Converts a tensor into an object that can be processed by the CPU/GPU.
'''

# Check out our model's parameters (a parameter is a value that the model sets itself)

# Set up Loss Function
loss_fn = nn.L1Loss()

# Setup an optimizer. Two parameters.
optimizer = torch.optim.SGD(params=houseModel.parameters(), # Model parameters you'd like to optimize.
                            lr=0.1 # The learning rate is how much it changes the parameter of a tensor value. e.g. if lr is set to 0.01, and the dataset is some random float 3.036, it'll change to 3.04.
                            # Shrinking the learning rate allows it to converge slower, yet prevents overfitting the original data.
                            )

# Building a training loop (and testing loop) in PyTorch.
'''
1. Loop through the data.
2. Forward pass. (Data moving through our model's 'forward()' functions) to make predicitons on data - also called forward propagation))
3. Calculate the loss (Compare forward pass predictions to ground truth labels)
4. Optimizer zero grad
5. Loss backward - Move backwarsd through the network to calculate the gradients of each of the parameters of our model with respect to the loss.
6. Optimizer step - Use the optimizer to adjust our model's parameters to try and improve the loss. (Gradient descent)

NOTE THAT THE GRADIENT AND THE BACK PROPAGATION USE A LOT OF MATH TO IMPROVE THE DATA FROM THEIR INITIAL VALUES TO THE EXEMPLARY DATABASE TO THE TEST DATABASE. The gradient is basically the slope of a graph on x and y axis.
'''

# An epoch is one loop through the data (It's a hyperparameter because we set it ourselves)
epochs = 500

# This whole thing can be replicated in a function.

# It should also be known that step 4 and 5 are specifically used as methods for training the model.

# Set the model to the GPU, otherwise use the CPU as defined prior.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Put data on the available device.
TrainingData = TrainingData.to(device)
IdealTrainingData = IdealTrainingData.to(device)
TestingData = TestingData.to(device)
IdealTestingData = IdealTestingData.to(device)

houseModel.to(device)

print(f"Now using device: {device} \n")

#foundationData = foundationData.to(device)

# Empty loss lists to track values
trainingLossVals = []
testingLossVals = []
epochCount = []

# 1. Loop through the data
for epoch in range(epochs): # Pass the data through the model for a number of epochs (e.g. 100 for 100 passes of the data)
    # Set the model to training mode
    houseModel.train() # Train mode in PyTorch sets all parameters that require gradients to require gradients.

    # 2. Forward pass from dataset "y_pred" train data using the forward() method inside.
    # We learned patents on the training data to evaluate our model on the test data.
    idealPrediction = houseModel(TrainingData)

    # 3. Calculate the loss from the model's predictions to the true values.
    loss = loss_fn(idealPrediction, IdealTrainingData) # Identifying the difference of error. Prediction first, training second.

    # 4. Zero the gradients of the optimizer (they accumulate by default)
    optimizer.zero_grad()
    
    # 5. Perform backpropagation on the loss with respect to the parameters of the model.
    loss.backward()

    # 6. Step the optimizer (perform gradient descent) (Makes some calculations how it should adjust the model parameters with regards to the backpropagation of the loss)
    optimizer.step() # By default how the optimizer changes will accumulate through the loop so.. we have to zero them above in step 3 for the next iteration of the loop.

    ### Testing Portion

    houseModel.eval() # Turns off different settings in the model not needed for evaluation/testing such as gradient tracking.
   # with torch.inference_mode(): # Turns off gradient tracking.

    with torch.inference_mode():
       # 1. Forward pass on test data through the model. (Calculate from start to finish for the layering of the values.)
       TestPrediction = houseModel(TestingData)

       # 2. Calculate loss on test data from TestPrediction to what it should be (Based on IdealTrainingData)
       testingLoss = loss_fn(TestPrediction, IdealTestingData)
    #plot_predictions(predictions=y_preds_new);

    # Printing out the status of what's happening to the data.
    if epoch % 10 == 0:
        epochCount.append(epoch) # Add the current count.
        trainingLossVals.append(loss.detach().numpy())
        testingLossVals.append(testingLoss.detach().numpy())
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {testingLoss} ") # MAE = Mean Absolute Error.

houseModel.state_dict()

## Making predictions with our trained model.

# Evaluation mode.
houseModel.eval()

with torch.inference_mode():
    housePredictions = houseModel(TestingData)

print(f"Printing the graph: \n")

# Put the data on the CPU and plot it.
plotChart = gP.pythonPlotting(TrainingData, IdealTrainingData, TestingData, IdealTestingData)

print(f"The size of housePredictions: {housePredictions.shape} \n")

plotChart.plottingChart(input_predictions=housePredictions.cpu())

#plotChart.plottingChart.plt.show()

## Saving our PyTorch model
modelPath = Path("models")

# Create models directory
modelPath.mkdir(parents=True, exist_ok=True) # Second parameter will check if the directory already exists it won't error out if it already exists.

# Create Model Save Path.
modelName = "01_pytorch_workflow_model_1.pth" # pth = Pytorch extension.
modelSavePath = modelPath / modelName

print(f"The model has decided these stats are the most applicable: \n {houseModel.state_dict()}")

print(f"The model will be saved at: {modelSavePath}")
torch.save(obj=houseModel.state_dict(), f=modelSavePath) # state_dict contains all the models trained/associated parameters and what state they're in.