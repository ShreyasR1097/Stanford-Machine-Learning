function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cmatrix  =  [ 0.01 0.03 0.1 0.3 1 3 10 30];
sigmamatrix = [ 0.01 0.03 0.1 0.3 1 3 10 30];
errormatrix = zeros(64,3);
k = 1;
for i=1:8
    for j=1:8
        model= svmTrain(X, y, Cmatrix(i), @(x1, x2) gaussianKernel(x1, x2, sigmamatrix(j)));
         predictions = svmPredict(model, Xval);
         errormatrix(k,1) = Cmatrix(i);
         errormatrix(k,2) = sigmamatrix(j);
         errormatrix(k,3) = mean(double(predictions ~= yval));
         k = k+1;
    end
end

[M,I] = min(errormatrix);
z = I(3);
C = errormatrix(z,1);
sigma = errormatrix(z,2);


% =========================================================================

end
