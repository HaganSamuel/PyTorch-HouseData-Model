import torch
from torch import nn # The foundation of all learning models.
from torch import optim # A package implementing various optimization algorithms.
from pathlib import Path

from os import path
import pandas as pd

numOfVars = 9 # Number of variables the model should pick up when deciding what should be the ideal prediction.

# Our ideal weight and bias parameters that we are attempting to reach.
weight = 0.4
bias = 0.9

# Model Properties
epochs = 4000
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a random seed for generating random numbers.
torch.manual_seed(42)

class MultipleLinearRegression(nn.Module):
    def __init__(self, inputDimension: int): # input_dim represents the number of independent variables.
        super().__init__()

        # Initialize model parameters for weight and bias.
        self.weights = nn.Parameter(torch.rand(inputDimension, 1, requires_grad=True, dtype=torch.float)) # Initialized the shape with the first two parameters (inpu_dim, 1) to account for multiple variables.
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights + self.bias # Uses matrix multiplication with "@" for the input and the weights parameter.

# Create train/test data split.
dataFile = pd.read_csv('MultiRegressionData.csv')
houseTest = torch.tensor(dataFile.to_numpy(), dtype=torch.float)
MultiDataVariant = (houseTest @ torch.full((numOfVars, 1), weight)) + bias

# Separating the training data from the study data by 80% and the testing data by 20%.
splitTrainingData = int(0.8*len(houseTest))

# Given Data
trainingData = houseTest[:splitTrainingData]
testingData = houseTest[splitTrainingData:]

# The data goal to reach.
IdealTrainingData = MultiDataVariant[:splitTrainingData] # Colin ":" positioning determines the first or last of the values based on the number value fo splitTrainingData.
IdealTestingData = MultiDataVariant[splitTrainingData:] # These datasets are used to test and learn upon the training data.

multiHouseModel = MultipleLinearRegression(inputDimension = numOfVars)

# Setting up loss function
loss_fn = nn.MSELoss()

# Setup an optimizer. Two parameters.
optimizer = torch.optim.SGD(params=multiHouseModel.parameters(), # Model parameters you'd like to optimize.
                            lr=0.00001 # The learning rate is how much it changes the parameter of a tensor value. e.g. if lr is set to 0.01, and the dataset is some random float 3.036, it'll change to 3.04.
                            # Shrinking the learning rate allows it to converge slower, yet prevents overfitting the original data.
                            )

# Put data on the available device.
trainingData = trainingData.to(device)
testingData = testingData.to(device)
IdealTrainingData = IdealTrainingData.to(device)
IdealTestingData = IdealTestingData.to(device)
multiHouseModel.to(device)

# Empty loss lists to track values
trainingLossVals = []
testingLossVals = []
epochCount = []

# 1. Loop through the data
for epoch in range(epochs): # Pass the data through the model for a number of epochs (e.g. 100 for 100 passes of the data)
    # Set the model to training mode
    multiHouseModel.train() # Train mode in PyTorch sets all parameters that require gradients to require gradients.

    # 2. Forward pass from dataset "y_pred" train data using the forward() method inside.
    # We learned patents on the training data to evaluate our model on the test data.
    idealPrediction = multiHouseModel(trainingData)

    # 3. Calculate the loss from the model's predictions to the true values.
    loss = loss_fn(idealPrediction, IdealTrainingData) # Identifying the difference of error. Prediction first, training second.

    # 4. Zero the gradients of the optimizer (they accumulate by default)
    optimizer.zero_grad()
    
    # 5. Perform backpropagation on the loss with respect to the parameters of the model.
    loss.backward()

    # 6. Step the optimizer (perform gradient descent) (Makes some calculations how it should adjust the model parameters with regards to the backpropagation of the loss)
    optimizer.step() # By default how the optimizer changes will accumulate through the loop so.. we have to zero them above in step 3 for the next iteration of the loop.

    ### Testing Portion

    multiHouseModel.eval() # Turns off different settings in the model not needed for evaluation/testing such as gradient tracking.
   # with torch.inference_mode(): # Turns off gradient tracking.

    with torch.inference_mode():
       # 1. Forward pass on test data through the model. (Calculate from start to finish for the layering of the values.)
       TestPrediction = multiHouseModel(testingData)

       # 2. Calculate loss on test data from TestPrediction to what it should be (Based on IdealTrainingData)
       testingLoss = loss_fn(TestPrediction, IdealTestingData)
    #plot_predictions(predictions=y_preds_new);

    # Printing out the status of what's happening to the data.
    if epoch % 10 == 0:
        epochCount.append(epoch) # Add the current count.
        trainingLossVals.append(loss.detach().numpy())
        testingLossVals.append(testingLoss.detach().numpy())
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {testingLoss} ") # MAE = Mean Absolute Error.

## Making predictions with our trained model.

# Evaluation mode.
multiHouseModel.eval()

with torch.inference_mode():
    housePredictions = multiHouseModel(testingData)

print(f"Model stats: {multiHouseModel.state_dict()} \n")