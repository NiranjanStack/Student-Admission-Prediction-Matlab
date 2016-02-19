% Computing sigmoid function
% A sigmoid function is a bounded differentiable real function 
% that is defined for all real input values and has a positive derivative at each point
% J = SIGMOID(z) computes the sigmoid of z.

function g = sigmoid(z)
g = zeros(size(z));

% Computing the sigmoid of each value of z

f=@(h) 1 ./ (1+exp(-h));
g=f(z);

end
