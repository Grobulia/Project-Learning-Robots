# Model-Agnostic Meta-Learning for Abstract Reasoning Tasks
This project was done  for the Learning Robots module at the University of Ulm. It is a result of the collective effort by Ksenia Vinogradova and Rohan Asthana.

## Introduction
Meta-learning in machine learning refers to learning algorithms that learn from other learning algorithms. The goal of a meta-learning model is to have one learning rule that can be easily updated so that it can be extrapolated to a new task. There is a variety of meta-learning methods that achieve that. In this project, we used a specific type of meta-learning called Model-Agnostic Meta-Learning, or MAML for short. We investigated whether MAML would show a better performance than other methods on one of the tasks in the Kaggle Abstraction and Reasoning Corpus challenge (ARC). The aim of the challenge is to test the capability of different AI approaches to solve problems that require abstract reasoning skills characteristic to humans. The corpus consists of 400 tasks, each of which containing 3-5 problem-answer pairs in the training set. The challenge is considered to be one of the most difficult of its kind and almost a thousand submissions have been made involving a variety of AI approaches. Of those submissions, we chose two for comparison with the MAML - a convolutional neural network and a decision tree.

We trained each of the three algorithms mentioned above on the same items of the filling task in the ARC dataset and then tested the trained models on random and manually generated task items. The details of each implementation and testing would be described in the following sections.

## Model-Agnostic Meta-Learning (MAML)
The MAML (Finn, Abbeel, and Levin, 2017) learns a generalised “internal representation” of the tasks that can be easily adapted to a new task by adjusting parameters a little with the help of gradient descent. This is usually achieved by having a dataset consisting of several subsets, each of which is a different task with some examples. In this way, the network learns to find the general pattern common to the different tasks in the set and fine-tune it to each new task.

For the implementation of MAML, we used the code provided in our course material on Moodle and modified it to fit our task. As the ARC problem involves recognizing patterns, we used the Regression MAML model including 2 Conv2D layers each with ReLU activation function. The input is of shape (10,10,1) and the Convolutional layers contain 50 and 100 neurons respectively followed by an output Dense layer with 10 neurons. The loss function used was Mean Squared Error (MSE).

## CNN
For CNN we used the existing implementation by Tarun Paparaju published on Kaggle. The approach uses a CNN model that accepts a 2D input and returns a 2D output. It consists of two convolutional layers followed by two max-pooling layers, a dense layer and the output layer that outputs softmax probabilities for each of the pixels (see figure 1). The softmax probabilities are then converted to the final output matrix with the use of argmax and resize functions. 

## Decision Tree
For the decision tree, another existing implementation (Siddhartha, Kaggle: meaninglesslives) was used. The tasks were first flattened and each pixel was used as an observation to be classified. To mitigate the loss of global structural information (i.e. the position of cells with relation to each other), additional features containing more global information, such as Moore neighbours and number of unique colours in column/row, were included. The number of features and Moore neighbours was manipulated and, through trial-and-error a set of 13 features and 6 neighbours were found to be the most optimal.

## Training
### Preparing Input
We used a single task JSON file from the ARC dataset that corresponds to the filling color task (Task 00d62c1b) as the input to all our models. The task file contains 6 pairs of input and output. We extract all the pairs with size 10x10. The values in these 10x10 matrices contain the numbers 0 to 9 depending on the color that particular cell has. In the filling color task, only 3 colors are used which are black, green and yellow.The final array is of size (3,2,10,10,1). A sample input- output pair has been demonstrated in Fig 2.

### Training MAML
Each input-output pair of the task is fed into the Regression model which outputs a 10x10 output matrix for a total of 50 epochs. This model output matrix is then compared with the correct output and MSE loss is calculated. After the calculation of loss, we calculate the gradient of the loss. Then we make a copy of the model and update the kernels and biases of all the layers using the gradients calculated. We then calculate the test loss using the copied model and calculate the gradients of the test loss. Finally, we apply those gradients to our Adam optimizer, calculate the mean of the losses and display the average loss. This average loss is used to update the meta-learning model. In this way, the algorithm learns to initialise hyper-parameters in such a way that the initialisation is close to the actual hyper-parameters by which the model converges

### Training the CNN
For training the CNN model, the 10x10 data was fed into the network which outputted a 10x10 matrix as a result. The learning rate of 0.01 and a batch size of 128 were used. The network was trained for 50 epochs.


### Training the Decision Tree
For training the Decision Tree, we used a Gradient Boosted Decision Tree using the xgboost library. Each cell data was fed into the tree as an observation. The number of features used was 13 which included cell position, cell colour, Moore neighbours and others. The number of local neighbours used were 8. Number of boosting rounds used for the algorithm were 25.

## Evaluation
In order to evaluate and compare the performance of the MAML, CNN and the Decision Tree, we chose the simplest accuracy measure that could be applied to all of them - the number of matching cells between their output and the ‘correct’ output. For the outputs of MAML and CNN the outputs were rounded up to have integer matrices.

Two evaluation sets, 8 items each with dimensions 10x10, were generated: one was a set of items that were manually created to resemble the training items, whereas the other one was randomly generated. For the second set, the ‘correct’ answers, i.e. matrices with correctly coloured cells, were created through manual editing of the randomly generated matrices. They can be found in the respective folders.