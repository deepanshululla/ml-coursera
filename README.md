### Programming Assignment Solutions to Coursera MOOC on Machine learning

#### Assignment 1: Linear regression

In this assigment we needed to calculate for multivariable linear regression

1) gradient descent
2) Cost function  
3) Feature normalization

#### Assignment 2: Logistic regression

In this assignment for logistic regression

1) the first step was to calculate the sigmoid function
2) Finding the cost function for unregularized Logistic Regression
3) Gradient descent for the unregularized Logistic Regression
4) Writing the prediction function which predicts values based on X*theta >=0.5 or not
5) Finding the cost function for regularized Logistic Regression
6) Gradient descent for the regularized Logistic Regression



#### Assignment 3: Neural networks and multiclassification

In this assignment, we mostly used feedforward 
1) the first step was to find logistic regression cost function and gradient which was same as in
assignmnet 2
2) The second task was to create a oneVsAll classifier where we computer the value of all_thetas by passing them into fmincg function(already implemented).
3) The thirs step was to write a prediction function for one vs all which was a task to create a function PREDICTONEVSALL(all_theta, X) will return a vector of predictions for each example in the matrix X
4) the final part was to create a function Predict which will return the label of an input given a trained neural network

#### Implement backpropagation algorithm for neural networks 

1) In this assignment, we first found out the feedforward parameters
for finding out the cost function. We then found out unregularized version of the same.

2) When training neural networks, it is important to randomly initialize the pa-rameters for symmetry breaking. One effective strategy for random initializa-tion is to randomly select values for Θ(l)uniformly in the range [-e,e]

3)  Then we calculate the error terms that measures how much that node was “responsible”for any errors in our output
After that we accululate the gradients and then Obtain the (unregularized) gradient for the neural network cost func-tion by dividing the accumulated gradients

After you have successfully implemeted the backpropagation algorithm, you will  add  regularization  to  the  gradient.

 Note that you shouldnotbe regularizing the first column of Θ(l)
whichis used for the bias term.
