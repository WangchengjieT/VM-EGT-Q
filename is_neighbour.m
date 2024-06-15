function [neighbeour_mat] = is_neighbour(pos, radius, self)
    % pos: 2d array, n rows, at least 2 columns
    % pos: pos(k, 1:2) represents the x, y-axis coordinates
    % return a n×n adjacency matrix, whose vertices are the row of pos(k, 1:2)
    % two points are adjacent iff their distance < radius
    % periodic boundary, side length of edge are specified in the distfun_xxx
    % self: 0：self inclusive， otherwise：self exclusive
    
%     for k = 1:(n - 1)
%         temp_mat_x = abs(pos((k + 1):n, 1) - pos(k, 1));
%         temp_mat_x = [temp_mat_x, ran(1) - temp_mat_x];
%         temp_mat_x = min(temp_mat_x, [], 2);
%         
%         temp_mat_y = abs(pos((k + 1):n, 2) - pos(k, 2));
%         temp_mat_y = [temp_mat_y, ran(2) - temp_mat_y];
%         temp_mat_y = min(temp_mat_y, [], 2);
% 
%         dist = vecnorm([temp_mat_x, temp_mat_y],2);
%     end
    
%     pos(:, 1:2)
    D = pdist(pos(:, 1:2), @distfun_period);
    neighbeour_mat = squareform(D) < radius;
    
    if ~self 
        neighbeour_mat = neighbeour_mat - diag(ones(size(pos, 1), 1));
    end
end
% 
% [2.7118    0.3995;
%     1.2828    1.2687;
%     1.4000    2.2117;
%     1.8855    2.4222;
%     0.5856    2.4693;
%     0.5631    0.0499;
%     0.4806    0.5445;
%     2.4802    0.8838;
%     0.3517    0.3177]