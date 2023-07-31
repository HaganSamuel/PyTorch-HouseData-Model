import matplotlib.pyplot as plt
import torch

class pythonPlotting:
    # For some reason IdealTestingData and IdealTrainingData are tuples. As well as testingDataVal.
    def __init__(self, TrainingData, IdealTrainingData, TestingData, IdealTestingData):
        self.mainData = TrainingData
        self.idealTrainingData = IdealTrainingData
        self.testingDataVal = TestingData
        self.idealTestingData = IdealTestingData
        self.predictions=None

        #print(f"The shape of TrainingData is: {TrainingData.shape}")

    def plottingChart(self, input_predictions):

        self.predictions = input_predictions

        xMainAxis = torch.gather(self.mainData, 1, index=self.gettingAxis(self.mainData, 0))#torch.tensor([[0],[0]]))
       
        yMainAxis = torch.gather(self.mainData, 1, index=self.gettingAxis(self.mainData, 1))#torch.tensor([[0],[0]]))

        xTestAxis = torch.gather(self.testingDataVal, 1, index=self.gettingAxis(self.testingDataVal, 0))

        yTestAxis = torch.gather(self.testingDataVal, 1, index=self.gettingAxis(self.testingDataVal, 1))

        # Plots the training data and the test data and then compares the predictions.
        plt.figure(figsize=(10,7)) # Dimensions of the graph.

        # Training data will be in blue.
        plt.scatter(xMainAxis, yMainAxis, c="b", s=4, label = "Training Data")

        print(f"\n The size of testingDataVal is: {self.testingDataVal.size()} \n Contents: {self.testingDataVal}")

        # Test data will be in green.
        plt.scatter(xTestAxis, yTestAxis, c="g", s=4, label="Testing data")

        # Predictions will be in red.
        if self.predictions is not None:
            print(f"The shape of predictions is: {self.predictions.shape} \n Contents: {self.predictions}")

            xPredictionsAxis = torch.gather(self.predictions, 1, index=self.gettingAxis(self.predictions, 0))

            yPredictionsAxis = torch.gather(self.predictions, 1, index=self.gettingAxis(self.predictions, 1))

            plt.scatter(xPredictionsAxis, yPredictionsAxis, c="r", s=4, label="Predictions")

        # Show the legend
        plt.legend(prop={"size" : 14});

        plt.savefig('my_plot.png')

        plt.show()

    def gettingAxis(self, dataUsed, zeroOrOne):
        match zeroOrOne:
            case 0:
                return torch.zeros(dataUsed.size(), dtype=torch.long)
            case 1:
                return torch.ones(dataUsed.size(), dtype=torch.long)
            case other:
                print(f"Error: zeroOrOne ({zeroOrOne}) is not a proper input.")
