x=load('data1.mat')
y=load('data2.mat')
z=load('data3.mat')
x=x.data;
y=y.data;
z=z.data;
data=[x ; y ; z]
save('data.mat', 'data');

A=zeros(length(data), 6);
A(:,1)=1;
A(:,2)=data(:,1);
A(:,3)=data(:,2)
A(:,4)=abs(data(:,1)).*data(:,2);
A(:,5)=abs(data(:,2)).*data(:,2);
A(:,6)=(data(:,1)).^3;
b=data(:,4)- data(:,3);
w=A\b;
err=b-A*w;