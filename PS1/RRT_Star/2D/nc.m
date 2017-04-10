nodesCoords = [];
for i = 1:length(nodes)
    nodesCoords = [nodesCoords; nodes(i).coord];
end
for i = 1:length(nodes)
    plot(nodesCoords(i,1), nodesCoords(i,2),'x')
    hold on;
end
% nodesCoords = smooth(nodesCoords);