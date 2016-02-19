% Student Admission Prediction using Logistic Regression

% Loading the data
% The first two columns contains the exam scores and the third column contains the result of admission.

data = load('dataset.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

%Plotting the data   

fprintf(['Plotting data with + indicates (y = 1) examples and o indicating (y = 0) points.\n']);

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nPress enter to continue.\n');
pause;


%  Computing the cost and gradient for logistic regression in costFunction.m
[m, n] = size(X);

X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Computing the initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

% Using a built-in function (fminunc) to find the optimal parameters theta.
% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  This function will return theta and the cost to obtain optimal theta
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

legend('Admitted', 'Not admitted')
hold off;

fprintf('\nPress enter to continue.\n');
pause;

% Using the logistic regression model to predict the probability that a student with score 80 on exam 1 and 
% score 35 on exam 2 will be admitted or not (predict.m)

prob = sigmoid([1 80 35] * theta);
fprintf(['Probability of getting admission for a score of 80 and 35 is %f\n\n'], prob);

% Computing accuracy on training set
p = predict(theta, X);

fprintf('Accuracy acheived is %f\n', mean(p == y) * 100);
