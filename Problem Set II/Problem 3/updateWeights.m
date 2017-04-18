function [Theta2, Theta1] = updateWeights(X, y, out, Theta1, Theta2, k, j, eta, v1)         

%updating the weight for the outermost layer
delta = y(1,j) - out;
Theta2(1,k) = Theta2(1,k) + eta*(delta*v1(k));
%Weight update for hidden-input layers
Theta1(1,k) = Theta1(1,k) + eta*(sigmoidGradient(v1(k))*(y(j) - out)*Theta2(1,k)*X(1,j));
Theta1(2,k) = Theta1(2,k) + eta*(sigmoidGradient(v1(k))*(y(j) - out)*Theta2(1,k)*X(2,j));
Theta1(3,k) = Theta1(3,k) + eta*(sigmoidGradient(v1(k))*(y(j) - out)*Theta2(1,k)*X(3,j));

end