s = [1,1,2,2,3,4,5,6];
t = [2,3,5,4,7,6,6,7];
weights = [50,50,20,10,50,20,10,20];
G = digraph(s,t,weights);
T = bfsearch(G,1,'allevents');
plot(G);