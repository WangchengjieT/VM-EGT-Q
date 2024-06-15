%% 
clc
clear

%% 
global L
L = 31.6;

v0 = 0.03;
% be = 1;
burn_in = 1000;
mc_step = 3;
B = 50;
rho = 1;
% al = 2;
eta = 0.5;

r = 1;

N = round(rho * L * L);
% N = 300;

% state_ini, state_temp, N rows, several columns
% The k-th row contains various about agent_k
% [1 x coordinate, 2 y coordinate, 3 theta in directions, 4 number of neighbors (self excluding),
%  5 average direction of neighbors (self including), 6 direction drift in next step]

% 1 x coordinate, 2 y coordinate, 3 theta in directions
state_all = [rand(N,2) .* L, rand(N, 1) .* 2 .* pi];

% 4 number of neighbors (self including)
adj_mat_temp = is_neighbour(state_all(:, 1:2), r, 1);
state_all(:, 4) = sum(adj_mat_temp, 2);
% 5 average direction of neighbors (self including)
veloc_temp = exp(state_all(:, 3) .* 1i);
veloc_temp = sum(adj_mat_temp .* veloc_temp.', 2);
state_all(:, 5) = mod(atan2(imag(veloc_temp), real(veloc_temp)), 2 * pi);
% 6 direction drift in next step (If cooperate, Unif (- eta, eta); if defect, Unif (0,2Pi))
state_all(:, 6) = unifrnd(-eta, eta, N, 1);

% axis tight manual
% v = VideoWriter('vicsek_origin_1.avi');
% open(v);

if 1
tic
for t = 2:(burn_in + mc_step + 1)

    % 1 x coordinate
    state_all(:, 1, t) = mod(state_all(:, 1, t - 1) + v0 .* cos(state_all(:, 3, t - 1)), L);
    % 2 y coordinate
    state_all(:, 2, t) = mod(state_all(:, 2, t - 1) + v0 .* sin(state_all(:, 3, t - 1)), L);
    % 3 theta in directions
    state_all(:, 3, t) = mod(state_all(:, 5, t - 1) + state_all(:, 6, t - 1), 2 * pi);
    
    % 4 number of neighbors (self including)
    adj_mat_temp = is_neighbour(state_all(:, 1:2, t), r, 1);
    state_all(:, 4, t) = sum(adj_mat_temp, 2);
    % 5 average direction of neighbors (self including)
    veloc_temp = exp(state_all(:, 3, t) .* 1i);
    veloc_temp = sum(adj_mat_temp .* veloc_temp.', 2);
    state_all(:, 5, t) = mod(atan2(imag(veloc_temp), real(veloc_temp)), 2 * pi);
    % 6 direction drift in next step (If cooperate, Unif (- eta, eta); if defect, Unif (0,2Pi))
    state_all(:, 6, t) = unifrnd(-eta, eta, N, 1);


    if ~mod(t, 100)
        disp(t)
    end

% %     if ~mod(t, 1000)
% %         plot_population(state_all(:, 1:4, t), [L, L], 0)
% %     end
%     
%     dx = cos(state_all(:, 3, t));
%     dy = sin(state_all(:, 3, t));
%     
% %     figure(1)
% %     
%     quiver(state_all(:, 1, t), state_all(:, 2, t), dx, dy, "b", "AutoScaleFactor", 0.5);
%     xlim([0, L]);  
%     ylim([0, L]);  

%     frame = getframe(gcf);
%     writeVideo(v, frame);
%   
end
toc
end

% close(v)











