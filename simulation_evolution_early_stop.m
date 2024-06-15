function [lev, state_all, Q_table_all] = simulation_evolution_early_stop(vicsek_para, qlearning_para, mc_para, step_more, if_plot)
    % simulation via vicsek model + evlutionary game: VM-EGT
    % 
    % parameter：
    % vicsek_para = [L, rho, N, v0, be, al, eta, r]
    % mc_para = [burn_in, mc_step, B]
    %
    % return：
    % level: array with (num_to_sim + 1) rows，2 columns
    % 2 columns denote：C, V_a
    %
    %
    %
    
    % parameter for Monte Carlo
    burn_in = mc_para(1); % burn in steps
    mc_step = mc_para(2); % length of the statinary sequence
    B = mc_para(3); % number to replicate the numerical simulation, to take average, reduce variance
    b_mc = mc_para(4); % mc step now
    group_ind = mc_para(5);
    num_to_sim = burn_in + mc_step;

    % parameter for Vicsek
    L = vicsek_para(1);
    rho = vicsek_para(2);
    N = vicsek_para(3);
    v0 = vicsek_para(4);
    be = vicsek_para(5);
    al = vicsek_para(6);
    eta = vicsek_para(7);
    r = vicsek_para(8);
    
%     Q_table_all = rand(N_Q + 1, N_Q, N);
%    Q_table_all = repmat((1:N_Q) / N_Q, N_Q + 1, 1, N);

    flag_temp = 1; % run or not

    %Stateuxxx is an array of N rows,
    %The k-th line records all the states of agent_k
    %[1x position, 2y position, theta in 3 directions, whether 4 cooperates with SL, number of neighbors (excluding oneself),
    %6 average directions (including oneself), 7 next direction drift, cooperation: Unif (- eta, eta), betrayal: Unif (0,2Pi),
    %8 local order parameters (excluding oneself, 0 if there are no neighbors), 9 communication costs, 10 payoff,
    %11 imitates the target number. If none of them imitate, the result is 0. 12 imitates the target and changes. If none of them imitate, the result is oneself,
    %13 Fermi prob, the probability of imitating oneself is 1/2, 14 whether to imitate, 15 strategic goals for the next step,
    
    %% 初始化
    % 1
    state_all = [rand(N,2) .* L, rand(N, 1) .* 2 .* pi];
    
    % 4
    state_all(:, 4) = randi([0, 1], N, 1);
    level = mean(state_all(:, 4));
    % 5
    adj_mat_temp = is_neighbour(state_all(:, 1:2), r, 0);
    state_all(:, 5) = sum(adj_mat_temp, 2);
    % 6
    veloc_temp = exp(state_all(:, 3) .* 1i);
    veloc_temp = sum((adj_mat_temp + eye(N)) .* veloc_temp.', 2);
    state_all(:, 6) = mod(atan2(imag(veloc_temp), real(veloc_temp)), 2 * pi);
    % 7
    state_all(:, 7) = unifrnd(-eta, eta, N, 1) + unifrnd(0, 2 * pi, N, 1) .* (~state_all(:, 4));
    % 8
    order_pair_temp = abs(exp(1i .* state_all(:, 3)) + exp(1i .* state_all(:, 3)')) ./ 2;
    nei_ind_temp = state_all(:, 5) ~= 0;
    state_all(:, 8) = 0;
    state_all(nei_ind_temp, 8) = sum(order_pair_temp(nei_ind_temp, :) .* adj_mat_temp(nei_ind_temp, :), 2) ./ state_all(nei_ind_temp, 5);
    % 9
    state_all(:, 9) = state_all(:, 4) .* r ./ L;
    % 10
    state_all(:, 10) = state_all(:, 8) - al .* state_all(:, 9);
    % 11
    state_all(:, 11) = neighbour_to_imitate(state_all(:, 5), adj_mat_temp);
    % 12
    state_all(:, 12) = state_all(:, 11) + (~state_all(:, 11)) .* (1:N)';
    % 13
    state_all(:, 13) = 1 ./ (1 + exp((state_all(:, 10) - state_all(state_all(:, 12), 10)) ./ be));
    % 14
    state_all(:, 14) = rand(N, 1) < state_all(:, 13);
    % 15
    state_all(:, 15) = state_all(:, 14) .*  state_all(:, 12) + (~state_all(:, 14)) .* (1:N)';
    

    
    t = 2;
    
    while flag_temp

        % 1
        state_all(:, 1, t) = mod(state_all(:, 1, t - 1) + v0 .* cos(state_all(:, 3, t - 1)), L);
        % 2
        state_all(:, 2, t) = mod(state_all(:, 2, t - 1) + v0 .* sin(state_all(:, 3, t - 1)), L);
        % 3
        state_all(:, 3, t) = mod(state_all(:, 6, t - 1) .* state_all(:, 4, t - 1) + state_all(:, 7, t - 1), 2 * pi);
    
        % 4
        state_all(:, 4, t) = state_all(state_all(:, 15, t - 1), 4, t - 1);
        % cooperation rate now
        level = mean(state_all(:, 4, t));
        if level == 1
            flag_temp = 0;
        elseif level == 0
            flag_temp = 0;
        elseif t >= num_to_sim + 1
            flag_temp = 0;
        end
        % 5
        adj_mat_temp = is_neighbour(state_all(:, 1:2, t), r, 0);
        state_all(:, 5, t) = sum(adj_mat_temp, 2);
        % 6 
        veloc_temp = exp(state_all(:, 3, t) .* 1i);
        % synchronization rate now
        veloc_temp = sum((adj_mat_temp + eye(N)) .* veloc_temp.', 2);
        state_all(:, 6, t) = mod(atan2(imag(veloc_temp), real(veloc_temp)), 2 * pi);
        % 7
        state_all(:, 7, t) = unifrnd(-eta, eta, N, 1) .* state_all(:, 4, t) + unifrnd(0, 2 * pi, N, 1) .* (~state_all(:, 4, t));
        % 8
        order_pair_temp = abs(exp(1i .* state_all(:, 3, t)) + exp(1i .* state_all(:, 3, t)')) ./ 2;
        nei_ind_temp = state_all(:, 5, t) ~= 0;
        state_all(:, 8, t) = 0;
        state_all(nei_ind_temp, 8, t) = sum(order_pair_temp(nei_ind_temp, :) .* adj_mat_temp(nei_ind_temp, :), 2) ./ state_all(nei_ind_temp, 5, t);
        % 9
        state_all(:, 9, t) = state_all(:, 4, t) .* r ./ L;
        % 10
        state_all(:, 10, t) = state_all(:, 8, t) - al .* state_all(:, 9, t);
        % 11
        state_all(:, 11, t) = neighbour_to_imitate(state_all(:, 5, t), adj_mat_temp);
        % 12
        state_all(:, 12, t) = state_all(:, 11, t) + (~state_all(:, 11, t)) .* (1:N)';
        % 13
        state_all(:, 13, t) = 1 ./ (1 + exp((state_all(:, 10, t) - state_all(state_all(:, 12, t), 10, t)) ./ be));
        % 14
        state_all(:, 14, t) = rand(N, 1) < state_all(:, 13, t);
        % 15
        state_all(:, 15, t) = state_all(:, 14, t) .*  state_all(:, 12, t) + (~state_all(:, 14, t)) .* (1:N)';
        
        t = t + 1;
    
    
        if ~mod(t, 200)
            fprintf('第%d组参数, alpha = %3.1f, r = %5.2f, eta = %3.1f, 实验: %d/%d, 进度: %3.1f \n', group_ind, al, r, eta, b_mc, B, t);
        end
        
    end

    % lev: state when stop
    lev = [t - 1, level];

end

%%
function [action_temp, ord] = pi_Q(q_table, state, payoff, adj_mat)
    % max step in Q-learning
    % 

    N = size(q_table);
    N_Q = N(2);
    N = N(3);
    action_temp = zeros(N, 1);
    ind = 1:N;
    ord = zeros(N, 1);

    for k = 1:N
        if state(k)
            num = min([state(k), N_Q]);
%             disp(state(k))
%             disp(num)
%             disp([state(k), N_Q])
%             disp(q_table(num, :, k))
%             disp(q_table(num, 1:num, k))
            [~, ord(k)] = max(q_table(num, 1:num, k));
            ind_temp = ind(adj_mat(:, k) == 1);
            [~, I] = sort(payoff(ind_temp), 'descend');
            action_temp(k) = ind_temp(I(ord(k)));
%         else
%             action_temp(k) = 0;
        end
    end
end

function [ntm, ord] = neighbour_to_imitate_Q(num_neighbour, adj_mat, payoff, q_table, epsilon_greedy)
    N = size(q_table);
    N_Q = N(2);
    N = N(3);

    prob = cat(2, zeros(N, 1), cumsum(adj_mat, 2));
    ntm_temp = rand(N, 1) .* num_neighbour;
    
    [action, ord] = pi_Q(q_table, num_neighbour, payoff, adj_mat);    
    I = rand(N, 1) > epsilon_greedy;

    % greedy action
    ord = I .* ord + ~I .* (floor(ntm_temp) + (num_neighbour ~= 0));
    ord = arrayfun(@(x) min([N_Q, x]), ord);

    ntm_temp = sum(prob < ntm_temp, 2);

    ntm = I .* action + ~I .* ntm_temp;
end





