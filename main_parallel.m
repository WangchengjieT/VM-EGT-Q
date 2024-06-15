%% parallel 

if 1 %  change to 1 just on the first run

clc;
% clear



al_range = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0];
% r_range = exp(linspace(log(0.2), log(25), 31));
r_range = [0.2, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 25];
et_range = [0.5, 1.0, 1.5, 1.7];


% parameter for Vicsek model and evolutionary game
L = 31.6; % side length
rho = 1; % density of agent
N = round(rho * L * L); % total number of agents

v0 = 0.1; % absolute velocity
be = 1; % beta: rational parameter， >0，The closer to 0, the more rational the agent becomes
% al = 1; % alpha，relative cost, >0，The larger the alpha value, the higher the communication cost
eta = 0.5; % noise level
% r = 1; % communication radius
% 
% vicsek_para = [L, rho, N, v0, be, al, eta, r];
% 
% paramter for Monte Carlo
burn_in = 20000; % burn in steps
mc_step = 1000; % length of stationary process
B = 50; % number to replicate the MC simulation, taking averge to reduce variance


% Q learning  parameter
c_prop = 0.1;
gam = 0.9;
epsilon_greedy = 0.05;
al_Q = 0.1;


al_len = length(al_range);
r_len = length(r_range);
et_len = length(et_range);
% result_df = zeros(al_len * r_len * B, 5);
% c_result = zeros(al_len * r_len * B, B);
% va_result = c_result;
% G_result = c_result;

para_mat = zeros(al_len * r_len * B, 4);
gr_mat = kron((1:(al_len * r_len))', ones(B, 1)); % 第几组参数
b_mat = repmat((1:B)', al_len * r_len, 1); % 第几次试验
al_mat = kron(al_range', ones(B * r_len, 1)); % alpha
r_mat = repmat(kron(r_range', ones(B, 1)), al_len, 1); % r
c_mat = zeros(al_len * r_len * B, 1);
va_mat = c_mat;
G_mat = c_mat;
end_step_mat = c_mat;
end_state_mat = c_mat;
end_step_evo_mat = c_mat;
end_state_evo_mat = c_mat;

% @repelem

end


%% VM-ETG-Q, alpha
initval = 1;
endval = 4800;
tic
parfor (k = initval:endval, 10)
    al = al_mat(k);
    r = r_mat(k);
    vicsek_para = [L, rho, N, v0, be, al, eta, r];
    b = b_mat(k);
    gr = gr_mat(k);
    N_Q = max([10, round(min([N, 50, rho * r * r * pi * 3]))]);    
    qlearning_para = [N_Q, gam, epsilon_greedy, al_Q];
    mc_para = [burn_in, mc_step, B, b, gr];
%      mc_para = [100, 100, B, b, gr];
    lev = simulation_Q(vicsek_para, qlearning_para, mc_para, 0);
    c_mat(k) = lev(1);
    va_mat(k) = lev(2);
    G_mat(k) = lev(3);
    fprintf(['第%d组参数, 实验: %d, alpha = %3.1f, r = %5.2f, eta = %3.1f, ' ...
        '<C> = %6.4f, <V_a> = %6.4f, G = %6.4f \n'], ...
        gr, b, al, r, eta, c_mat(k), va_mat(k), G_mat(k))
end
toc

time_now = clock;
fprintf("当前时间： %d/%d/%d %2.0f:%2.0f:%2.0f \n", time_now);

para_mat = [(1:4800).', gr_mat, b_mat, al_mat, r_mat, c_mat, va_mat, G_mat];
writematrix(para_mat(initval:endval, :), 'vicsek_reinforcement_result_fig1.csv', 'WriteMode','append');




%% VM-ETG
if 1
    gr_mat = kron((1:(al_len * r_len))', ones(B, 1)); % 第几组参数
    b_mat = repmat((1:B)', al_len * r_len, 1); % 第几次试验
    al_mat = kron(al_range', ones(B * r_len, 1)); % alpha
    r_mat = repmat(kron(r_range', ones(B, 1)), al_len, 1); % r
    et_mat = 0.5 * ones(B * al_len * r_len, 1); % alp
    c_mat = zeros(al_len * r_len * B, 1);
    va_mat = c_mat;
    st_mat = c_mat;
end

initval = 1;
endval = 4400;
tic
parfor (k = initval:endval, 10)
% for k = initval:endval
% for k = 1 + (5:95) .* 50
    al = al_mat(k);
    r = r_mat(k);
    et = et_mat(k);
    vicsek_para = [L, rho, N, v0, be, al, et, r];
    b = b_mat(k);
    gr = gr_mat(k);
    N_Q = max([10, round(min([N, 50, rho * r * r * pi * 3]))]);    
    qlearning_para = [N_Q, gam, epsilon_greedy, al_Q];
    mc_para = [burn_in, mc_step, B, b, gr];
%       mc_para = [1, 1000, B, b, gr];

    lev = simulation_evolution_early_stop(vicsek_para, qlearning_para, mc_para, 0);
    c_mat(k) = lev(2);
    va_mat(k) = lev(3);
    st_mat(k) = lev(1); 
    fprintf(['演化博弈, 试验：第%d组参数, 实验: %d, alpha = %3.1f, r = %5.2f, eta = %3.1f, ' ...
        '<C> = %6.4f, <V_a> = %6.4f, 统一时间 = %d \n'], ...
        gr, b, al, r, eta, c_mat(k), va_mat(k), st_mat(k))
end
toc

time_now = clock;
fprintf("当前时间： %d/%d/%d %2.0f:%2.0f:%2.0f \n", time_now);

para_mat = [(1:(al_len * r_len * B)).', gr_mat, b_mat, al_mat, r_mat, et_mat, c_mat, va_mat, st_mat];
writematrix(para_mat(initval:endval, :), 'vicsek_evolution_alpha.csv', 'WriteMode','append');








