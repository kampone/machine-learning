function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
X_temp = X;
regularization = 0;


% =============================================================
%J = (1/m)*sum(-y .* log (sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta)));
for i = 2 : size(theta)
   regularization = regularization + theta(i) ^ 2;
  end


J = (1/m)*sum(-y .* log (sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta))) + (lambda/(2*m)) * regularization; 

coef = sigmoid(X * theta) - y;
for i = 1 : size(X_temp, 1)
  X_temp(i, :) = X_temp(i, :) .* coef(i); 
  end
grad_temp = sum(X_temp)./m;
grad(1) = grad_temp(1);
for i = 2:size(grad)
    grad(i) = grad_temp(i) + lambda/m * theta(i);
  end

% =============================================================
end
