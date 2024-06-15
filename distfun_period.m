function [D2] = distfun_period(ZI, ZJ)
    % ZI: vector of size 1×n containing one oberservation
    % ZJ: vector of size m2×n containing several oberservations
    % distfun:  m2 can be any postive integer
    % D2: vector of size m2×1 containing the distance between ZI and ZJ(k, :)
    L = 31.6;
    
    temp_z = abs(ZJ - ZI);
    for k = 1:size(ZJ, 2)
        temp_z(:, k) = min([temp_z(:, k), L - temp_z(:, k)], [], 2);
    end
    D2 = vecnorm(temp_z, 2, 2);

end