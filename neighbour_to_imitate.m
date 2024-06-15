function [ntm] = neighbour_to_imitate(num_neighbour, adj_mat)
    % Determine the neighbor number that each agent imitates under the Fermi rule based on the state and adjacency matrix adj_mat
    % If there are no neighbors, return 0
   
    n = size(num_neighbour, 1);
    
    prob = cat(2, zeros(n, 1), cumsum(adj_mat, 2));
    ntm = rand(n, 1) .* num_neighbour;
    
    ntm = sum(prob < ntm, 2);
end