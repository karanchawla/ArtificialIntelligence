function sum = summation(X,Y,prior_mean,j)
sum = (X - prior_mean(1,j)).^2 + (Y - prior_mean(2,j)).^2;
end