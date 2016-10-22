function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    theta = theta - alpha .* delta(X,y,theta,m)';
    J_history(iter) = computeCost(X, y, theta);

end

end

function s = delta(X,y,theta,m)
  
h = X*theta;
s = zeros(1,size(X,2));
  for i = 1:m
    s = s + (h(i) - y(i)) * X(i, :);   
  end
  
  s = s./m;
  
endfunction

