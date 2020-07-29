# InvKinNet-Pytorch
Deep Neural Network that computes the inverse kinematics of a 5R robot arm
## Files
The .txt files contains computed training and validation data. You can make newones with the generate_data.py script.  training.py is the training script while old_traininig.py was made for an earlier release.
## How to use the script
1) Rename the data path in the training file
2 ) run the following command "python training.py --b BATCHSIZE --s NETWORKSIZE --n SESSIONNUMBER --lr LEARNINGRATE --wd WEIGHTDECAY
These arguments are optional. The size must be between 2 and 100 (you can use more if you have the ressources)
## The Network 
This network a auto-encoder like parafigm. The "encoder" part is the network we want to train. It takes a quaternion a input and outputs the different angles for the arm. The "decoder" part is a static forward kinematic network. 


## Results & limits 
You can obtain a precision of 1e-2m. However, this network cannot be used for trajectory assessment for the angles would follow a discontinuous curve.
