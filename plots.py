import os, json, re
import numpy as np
from matplotlib import pyplot as plt


dataDir = "./mlpModels"


## Lets first plot the loss v/s the DOF for the MLP
DOFs = []
TestLoss = []
for fileName in os.listdir(dataDir):
    filePath = os.path.join(dataDir, fileName)
    if not filePath.endswith(".json"):
        continue
    with open(filePath, 'r') as file:
        data = json.load(file)[0]
        DOFs.append(int(re.search(r'\d+', fileName).group()))
        TestLoss.append(data["FinalValLoss"])


plt.scatter(DOFs, TestLoss)
plt.xlabel("Degrees of Freedom (DOFs)")
plt.ylabel("Test Loss (MSE)")
plt.title("Test Loss vs. Degrees of Freedom for MLP")
plt.show()