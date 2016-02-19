%COSTFUNCTION is to compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, x, y) computes the cost using theta as the
%   parameter for logistic regression and the gradient of the cost w.r.t. to the parameters.

function [J, grad] = costFunction(theta, X, y)
% number of training examples
m = length(y);

J = 0;
grad = zeros(size(theta));

%Setting J to the cost
%Computing the partial derivatives and setting grad to the partial derivatives of the cost w.r.t. each parameter in theta

p=sigmoid(X*theta);
y1=-y'*log(p);
y2=(1-y')*log(1-p);
J=(1/m)*(y1-y2);
grad=(1/m)*((p-y)'*X);

end