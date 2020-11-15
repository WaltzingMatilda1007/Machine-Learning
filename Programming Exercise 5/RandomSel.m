function [error_train_av, error_val_av] = ...
                                RandomSel(X, y, Xval, yval, lambda, num_it)

m = size(X, 1);

% You need to return these values correctly
error_train_av = zeros(m, 1);
error_val_av = zeros(m, 1);
error_train = zeros(m, num_it);
error_val = zeros(m, num_it);

for n = 1:num_it
  rand_indices = randperm(m);
  for i = 1:m
    X_rand = X(rand_indices(1:i),:);
    y_rand = y(rand_indices(1:i));
    Xval_rand = Xval(rand_indices(1:i),:);
    yval_rand = yval(rand_indices(1:i));
    theta = trainLinearReg(X_rand, y_rand, lambda);
    error_train(i,n) = linearRegCostFunction(X_rand, y_rand, theta, 0);
    error_val(i,n) = linearRegCostFunction(Xval_rand, yval_rand, theta, 0);
  endfor
endfor
  error_train_av = mean(error_train,2);
  error_val_av = mean(error_val,2);
end
