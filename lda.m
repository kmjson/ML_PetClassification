%Linear discriminant analysis training error rate is 1.1%.
%Linear discriminant analysis test error rate is 20%.

%This function takes in a training data matrix Xtrain, training
%label vector ytrain and uses them to compute the cat
%and dog sample mean vectors as well as the sample covariance matrix 
%(which is assumed to be equal for cats and dogs). 
%It also takes in a data matrix Xrun and produces a vector of
%label guesses yguess, corresponding to the ML rule for
%jointly Gaussian vectors with different means and the same 
%covariance matrix.
function yguess = lda(Xtrain,ytrain,Xrun)

[avgcat avgdog] = average_pet(Xtrain,ytrain);
[rt ct] = size(Xtrain);
[rr cr] = size(Xrun);
yguess = zeros(rr,1);
cat = zeros(1,ct);
dog = zeros(1,ct);

for i = 1:rt
    if ytrain(i,1) == -1
        cat = [cat;Xtrain(i,:)];
    else
        dog = [dog;Xtrain(i,:)];   
    end
end

cat(1,:) = [];
dog(1,:) = [];

sum = ((size(dog,1)-1).*(cov(dog)) + (size(cat,1)-1).*(cov(cat)))./(rt-1);
diff = pinv(sum)*transpose(avgdog-avgcat);
LDA = (avgdog*pinv(sum)*avgdog')-(avgcat*pinv(sum)*avgcat');

for i = 1:rr
    if (2*Xrun(i,:)*diff) >= LDA
        yguess(i,1) = 1;
    else
        yguess(i,1) = -1;
    end
        
end

if (~iscolumn(yguess))
    error("yguess is not a column vector.")
elseif (length(yguess)~=size(Xrun,1))
    error("Length of yguess is not equal to the number of rows in Xrun.")
elseif (sum(unique(abs(ytrain))~=1))
    warning("Some elements in ytrain are not +1 or -1.")
elseif (sum(unique(abs(yguess))~=1))
    warning("Some elements in yguess are not +1 or -1.")
end