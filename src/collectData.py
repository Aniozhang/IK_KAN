from robot import RobotArm
import os, random
import numpy as np
from tqdm import tqdm

## Create the dataset folder.
datasetPath = "./dataset"
if not os.path.isdir(datasetPath): 
    os.makedirs(datasetPath)


def generate_random_list(n):
    result = []
    for _ in range(n):
        # Create a list with three zeros
        sublist = [0, 0, 0]
        # Choose a random index (0, 1, or 2) to set to 1
        random_index = random.randint(0, 2)
        sublist[random_index] = 1
        result.append(sublist)
    return result


## Max DOF of the robots
n = 50
## Number of data points for a specific robot arm.
m = 10000

## Go through all the available DOFs and create the dataset for each.
for dof in tqdm(range(4, n)):
    numJoints = dof
    ## Let all the llenghts be 1 unit
    jointLengths = [1] * dof

    ## Generate the list of joints with random axis for joints
    jointAxes = generate_random_list(dof)

    ## Set all joint limits to -180 to 180 for simplicity.
    jointLimits = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

    robot = RobotArm(numJoints, jointLengths, jointAxes, jointLimits)

    ## Create the dataset lists for this robot arm
    datasetX = []
    datasetY = []

    ## Go through each datapoint.
    for _ in range(m):

        ## Go through each DOF
        jointPositions = []
        
        jointPositions = np.random.uniform(-np.pi, np.pi, dof).tolist()

        position, orientation = robot.forward_kinematics(jointPositions)
        finalOrintation = []

        finalOrintation.extend(position[-1])
        finalOrintation.extend(orientation.tolist())
        
        ## Generate the datapoints.
        datasetX.append(jointPositions)
        datasetY.append(finalOrintation)


    datasetX = np.array(datasetX)
    datasetY = np.array(datasetY)
    

    dirNname = str(dof) + "_robot"
    dirPath = os.path.join(datasetPath, dirNname)
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)


    np.save(dirPath + "/datasetInput", datasetX)
    np.save(dirPath + "/datasetOutput", datasetY)