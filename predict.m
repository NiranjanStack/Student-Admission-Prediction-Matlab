% Predicting whether a student gets admission or not using learned 
%logistic regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

function p = predict(theta, X)

% Number of training examples
m = size(X, 1); 

p = zeros(m, 1);

% Making predictions using learned logistic regression parameters. 

f=@(n) round(n);
s = sigmoid(X * theta);
p = f(s);

end
