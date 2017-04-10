clearvars
close all 

rows = 5;
cols = 5;
numNodes = 15;

qStart.coord = [1 1];
qStart.parent = 0;
qStart.visited = 1;
% qStart.cost = 0;

qGoal.coord = [rows cols];
qGoal.parent = -1;
qGoal.visited = 0;
% qGoal.cost = 0;

gridWorld = zeros(rows,cols);
nodes(1) = qStart;
h = figure(1);
hold on

for i=1:numNodes
    qRand = [floor(rand(1)*cols)+1 floor(rand(1)*rows)+1];
    plot(qRand(1), qRand(2), 'x', 'Color',  [0 0.4470 0.7410])
    hold on;
    
    for j=1:length(nodes)
        if nodes(j).coord == qGoal.coord
            break
        end        
    end
    
    ndist = [];
    
    for j = 1:length(nodes)
        n = nodes(j);
        tmp = dist(n.coord, qRand);
        ndist = [ndist tmp];
    end
    
    [~, idx] = min(ndist);
    
    qNear = nodes(idx);
%     qNew.coord = qRand;
%     line([qNear.coord(1), qNew.coord(1)], [qNear.coord(2), qNew.coord(2)], 'Color', 'b', 'LineWidth', 2);
%     qNew.cost = dist(qNew.coord, qNear.coord) + qNear.cost;
    
%     qNearest = [];
%     r = 5;
%     neighborCount = 1;
%     for j = 1:1:length(nodes)
%         if dist(nodes(j).coord, qNew.coord) <= r
%             qNearest(neighborCount).coord = nodes(j).coord;
%             qNearest(neighborCount).cost = nodes(j).cost;
%             neighborCount = neighborCount+1;
%         end
%     end
%     
%     qMin = qNear;
%     CMin = qNew.cost;
    
%     for k = 1:1:length(qNearest)
%         if qNearest(k).cost + dist(qNearest(k).coord, qNew.coord) < CMin
%             qMin = qNearest(k);
%             CMin = qNearest(k).cost + dist(qNearest(k).coord, qNew.coord);
%             line([qMin.coord(1), qNew.coord(1)], [qMin.coord(2), qNew.coord(2)], 'Color', 'g');
%             hold on
%         end
%     end
    
    qNew.coord = qRand;
%     qNew.parent(1) = qNear.coord(1);
%     qNew.parent(2) = qNear.coord(2);
    qNew.visited = 1;
    qNew.parent = idx;
%     for j = 1:1:length(nodes)
%         if nodes(j).coord == qMin.coord
%             qNew.parent = j;
%         end
%     end
    
%     line([qNear.coord(1), qNew.coord(1)], [qNear.coord(2), qNew.coord(2)], 'Color', 'b', 'LineWidth', 1);
    hold on;
    nodes = [nodes qNew];
%     saveas(h,sprintf('FIG%d.png',i))
end

D = [];
for j = 1:1:length(nodes)
    tmpdist = dist(nodes(j).coord, qGoal.coord);
    D = [D tmpdist];
end

% Search backwards from goal to start to find the optimal least cost path
[val, idx] = min(D);
qFinal = nodes(idx);
qGoal.parent = idx;
qEnd = qGoal;
nodes = [nodes qGoal];
goalNodes = [];
while qEnd.parent ~= 0
    start = qEnd.parent;
    goalNodes = [goalNodes; qEnd.coord];
    line([qEnd.coord(1), nodes(start).coord(1)], [qEnd.coord(2), nodes(start).coord(2)], 'Color', 'r', 'LineWidth', 2);
    hold on
    qEnd = nodes(start);
end
