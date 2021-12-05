%For PCA-reduced data with dimension k = 50, linear discriminant analysis training error rate is 7.6%.
%For PCA-reduced data with dimension k = 50, linear discriminant test error rate is 9.5%.
%For PCA-reduced data with dimension k = 50, perceptron training error rate is 7.6%.
%For PCA-reduced data with dimension k = 50, perceptron test error rate is 9.5%.
%For PCA-reduced data with dimension k = 100, linear discriminant analysis training error rate is 6.3%.
%For PCA-reduced data with dimension k = 100, linear discriminant test error rate is 9.3%.
%For PCA-reduced data with dimension k = 100, perceptron training error rate is 6.3%.
%For PCA-reduced data with dimension k = 100, perceptron test error rate is 9.3%.
%For PCA-reduced data with dimension k = 200, linear discriminant analysis training error rate is 5.4%.
%For PCA-reduced data with dimension k = 200, linear discriminant test error rate is 8.5%.
%For PCA-reduced data with dimension k = 200, perceptron training error rate is 5.4%.
%For PCA-reduced data with dimension k = 200, perceptron test error rate is 8.5%.
%For PCA-reduced data with dimension k = 400, linear discriminant analysis training error rate is 3.5%.
%For PCA-reduced data with dimension k = 400, linear discriminant test error rate is 8.3%.
%For PCA-reduced data with dimension k = 400, perceptron training error rate is 3.5%.
%For PCA-reduced data with dimension k = 400, perceptron test error rate is 8.3%.

%This function takes in a training data matrix Xtrain and uses
%it to compute the PCA basis and a sample mean vector. 
%It also takes in a test data matrix Xtest and a dimension k. 
%It first centers the data matrices Xtrain and Xtest by subtracting the
%Xtrain sample mean vector from each of their rows. It then uses the 
%top-k vectors in the PCA basis to project the centered Xtrain and Xtest
%data matrices into a k-dimensional space, and outputs
%the resulting data matrices as Xtrain_reduced and Xtest_reduced.
function [Xtrain_reduced Xtest_reduced] = reduce_data(Xtrain,Xtest,k)

V = pca(Xtrain);
[rv cv] = size(V);
Vk = V(1:rv,1:k);
avg_train = sum(Xtrain,1)/size(Xtrain,1);

test = ones(size(Xtest,1),1);
train = ones(size(Xtrain,1),1);
Xtest_cen = Xtest-(test*avg_train);
Xtrain_cen = Xtrain-(train*avg_train);

Xtest_reduced = Xtest_cen*Vk;
Xtrain_reduced = Xtrain_cen*Vk;

%Your code should go above this line.
if (size(Xtrain_reduced,1)~=size(Xtrain,1)) 
    error("The number of rows in Xtrain_reduced is not the same as the number of rows in Xtrain.")
elseif (size(Xtest_reduced,1)~=size(Xtest,1)) 
    error("The number of rows in Xtest_reduced is not the same as the number of rows in Xtest.")
elseif (size(Xtrain_reduced,2)~=k) 
    error("The number of columns in Xtrain_reduced is not equal to k.")
elseif (size(Xtest_reduced,2)~=k) 
    error("The number of columns in Xtest_reduced is not equal to k.")
end
