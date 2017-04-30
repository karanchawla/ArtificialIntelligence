function post_mean = computeCentroids(sum,X,Y)
        % Go over every example, find its closest centroid, and store
        %the index inside idx at the appropriate location.
        [~, idx] = sort(sum);
        sum_sorted = idx(1,:);
        posterior_sum = zeros(2,5);
        ctr = ones(1,5);
        %go over each data point
        for j=1:500
            for t = 1:5
                %check which idx it has been assigned to
                if sum_sorted(j)==t
%                   Go over every centroid and compute mean of all points that
%                   belong to it.
                    posterior_sum(1,t) = posterior_sum(1,t) + X(j);
                    posterior_sum(2,t) = posterior_sum(2,t) + Y(j);
                    ctr(1,t) = ctr(1,t)+1;
                end
            end
        end
        post_mean(1,:) = posterior_sum(1,:)./ctr;
        post_mean(2,:) = posterior_sum(2,:)./ctr;
end