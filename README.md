# FuzzyLogic-IK
This project presents a novel and efficient method for calculating inverse kinematic solutions on small devices using KANs


## TO RUN
### Generate the dataset
* python collectData.py

### Train the KAN models
* python kanModel.py

### Train the MLP models
* python baselineMLP.py

  
## TO DO
* Create the simulation and collect data (Done)
* Train the KAN models for all the robot configuration (Done)
* Extract the mathematical equations from the KAN model (Anio)
* Compare with other methods such as MLP based Fuzzy Logic (Done)
* Extract the weights from the pytorch MLP model and try to execute them on a micro controller. (Anio)
* Extract the parameters from KAN and try to execute them on a micro controller. (Ashwin)
* Evalute the time and space complexity of each method (Ashwin)
* Write paper (Ashwin + Anio)

