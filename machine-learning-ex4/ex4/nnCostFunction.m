function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5).
% This is most easily done using an eye() matrix of size num_labels, with vectorized indexing by 'y'.
% A useful variable name would be "y_matrix", as this
Y=eye(num_labels)(y,:);


% Perform the forward propagation:

% a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
%z2 equals the product of a1 and Theta1
%a2 is the result of passing z2 through g()
%Then add a column of bias units to a2 (as the first column).
%NOTE: Be sure you DON'T add the bias units as a new row of Theta.
%z3 equals the product of a2 and Theta2
%a3 is the result of passing z3 through g()

a1=[ones(m,1) X];
z2=a1*Theta1';
a2=sigmoid(z2);
a2=[ones(size(z2,1),1) a2];
z3=a2*Theta2';
a3=sigmoid(z3);
h=a3;

% Unregularized cost function
J_unregularized=sum(sum(-Y.*log(h)-(1-Y).*log(1-h),2))/m;

% calculte penalty
p = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));

% Regularized cost function
J=J_unregularized+(lambda*p)/(2*m);

%Backpropagation

% 2: δ3 or d3 is the difference between a3 and the y_matrix. 
%The dimensions are the same as both, (m x r).

%3: z2 comes from the forward propagation process -
% it's the product of a1 and Theta1, prior to applying the sigmoid() 
%function. Dimensions are (m x n) ⋅ (n x h) --> (m x h). 
%In step 4, you're going to need the sigmoid gradient of z2. 
%From ex4.pdf section 2.1, we know that if u = sigmoid(z2), 
%then sigmoidGradient(z2) = u .* (1-u).

%4: δ2 or d2 is tricky. 
%It uses the (:,2:end) columns of Theta2. 
%d2 is the product of d3 and Theta2 (without the first column), 
%then multiplied element-wise by the sigmoid gradient of z2. 
%The size is (m x r) ⋅ (r x h) --> (m x h). 
%The size is the same as z2.
d3=a3.-Y;
d2=(d3*Theta2(:,2:end)).*sigmoidGradient(z2);

% Note: Excluding the first column of Theta2 is 
% because the hidden layer bias unit has no connection 
% to the input layer - so we do not use backpropagation 
% for it. See Figure 3 in ex4.pdf for a diagram showing this.


% accumulate gradients
% 5: Δ1 or Delta1 is the product of d2 and a1. 
% The size is (h x m) ⋅ (m x n) --> (h x n)
Delta1=d2'*a1;

% 6: Δ2 or Delta2 is the product of d3 and a2.
%  The size is (r x m) ⋅ (m x [h+1]) --> (r x [h+1])
Delta2=d3'*a2;
% 7: Theta1_grad and Theta2_grad are the same size as their 
% respective Deltas, just scaled by 1/m.

Theta1_grad=Delta1./m;
Theta2_grad=Delta2./m;
% Accuracy 95%


% calculate regularized gradient
% Note that you shouldnotbe regularizing the first column of Θ(l)
% whichis used for the bias term
p1 = [zeros(size(Theta1, 1), 1) (lambda/m)*Theta1(:, 2:end)];
p2 = [zeros(size(Theta2, 1), 1) (lambda/m)* Theta2(:, 2:end)];
% Accuracy 96% for 50 iterations and 99.6 for 400
% or use below method although above method has 1% more acuracy
% set the first column of Theta1 and Theta2 to all-zeros
% Theta1(:,1)=0;
% Theta2(:,1)=0;
% Scale each Theta matrix by λ/m. Use enough parenthesis so the operation is correct.
% p1=(lambda/m)*Theta1;
% p2=(lambda/m)*Theta2;
% If your implementation is correct, you should 
% see a reportedtraining  accuracy  of  about  95.3%  (this  may  vary  by  about  1%  due  to  therandom  initialization).   It  is  possible  to  get  higher  training  accuracies  bytraining  the  neural  network  for  more  iterations.   We  encourage  you  to  trytraining the neural network for more iterations 
% (e.g., setMaxIterto 400) andalso vary the regularization parameterλ.

Theta1_grad = Theta1_grad+ p1;
Theta2_grad = Theta2_grad + p2;
% Accuracy 95%














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
