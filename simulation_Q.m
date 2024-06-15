function [lev, state_all, Q_table_all] = simulation_Q(vicsek_para, qlearning_para, mc_para, if_plot)
    % simulation via vicsek model + evlutionary game + Q learning: VM-EGT-Q
    % on policy
    %
    % parameter：
    % vicsek_para = [L, rho, N, v0, be, al, eta, r]
    % mc_para = [burn_in, mc_step, B]
    % qlearning_para = [gam, epsilon_greedy];
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
    
    % parameter for Q-learning
    N_Q = qlearning_para(1);
    gam = qlearning_para(2);
    epsilon_greedy = qlearning_para(3);
    al_Q = qlearning_para(4);

    Q_table_all = repmat((1:N_Q) / N_Q, N_Q + 1, 1, N);
    level = zeros(num_to_sim + 1, 2);
    
    %Stateuxxx is an array of N rows,
    %The k-th line records all the states of agent_k
    %[1x position, 2y position, theta in 3 directions, whether 4 cooperates with SL, number of neighbors (excluding oneself),
    %6 average directions (including oneself), 7 next direction drift, cooperation: Unif (- eta, eta), betrayal: Unif (0,2Pi),
    %8 local order parameters (excluding oneself, 0 if there are no neighbors), 9 communication costs, 10 payoff,
    %11 imitates the target number. If none of them imitate, the result is 0. 12 imitates the target and changes. If none of them imitate, the result is oneself,
    %13 Fermi prob, the probability of imitating oneself is 1/2, 14 whether to imitate, 15 strategic goals for the next step,
    %16 Status: min (number of neighbors, N_Q), 17action]
    
    %% initialization
    % 1, 2, 3
    state_all = [rand(N,2) .* L, rand(N, 1) .* 2 .* pi];
    
    % 4
    state_all(:, 4) = randi([0, 1], N, 1);
    level(1, 1) = mean(state_all(:, 4));
    % 5
    adj_mat_temp = is_neighbour(state_all(:, 1:2), r, 0);
    state_all(:, 5) = sum(adj_mat_temp, 2);
    % 16
    state_all(:, 16) = arrayfun(@(x) min(N_Q, x), state_all(:, 5));
    % 6
    veloc_temp = exp(state_all(:, 3) .* 1i);
    level(1, 2) = abs(mean(veloc_temp));
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
    % 17
    [state_all(:, 11), state_all(:, 17)] = neighbour_to_imitate_Q(state_all(:, 5), adj_mat_temp, state_all(:, 10), Q_table_all, epsilon_greedy);
    % 12
    state_all(:, 12) = state_all(:, 11) + (~state_all(:, 11)) .* (1:N)';
    % 13
    state_all(:, 13) = 1 ./ (1 + exp((state_all(:, 10) - state_all(state_all(:, 12), 10)) ./ be));
    % 14
    state_all(:, 14) = rand(N, 1) < state_all(:, 13);
    % 15
    state_all(:, 15) = state_all(:, 14) .*  state_all(:, 12) + (~state_all(:, 14)) .* (1:N)';
    
    
    for t = 2:(num_to_sim + 1)

        % 1
        state_all(:, 1, t) = mod(state_all(:, 1, t - 1) + v0 .* cos(state_all(:, 3, t - 1)), L);
        % 2
        state_all(:, 2, t) = mod(state_all(:, 2, t - 1) + v0 .* sin(state_all(:, 3, t - 1)), L);
        % 3
        state_all(:, 3, t) = mod(state_all(:, 6, t - 1) .* state_all(:, 4, t - 1) + state_all(:, 7, t - 1), 2 * pi);
    
        % 4
        state_all(:, 4, t) = state_all(state_all(:, 15, t - 1), 4, t - 1);
        % cooperation rate now
        level(t, 1) = mean(state_all(:, 4, t));
        % 5
        adj_mat_temp = is_neighbour(state_all(:, 1:2, t), r, 0);
        state_all(:, 5, t) = sum(adj_mat_temp, 2);
        % 16
        state_all(:, 16, t) = arrayfun(@(x) min(N_Q, x), state_all(:, 5, t));
        % 6
        veloc_temp = exp(state_all(:, 3, t) .* 1i);
        % synchronization rate now
        level(t, 2) = abs(mean(veloc_temp));
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
        
        % updating q table
        for k = 1:N
%             disp([state_all(k, 16, t - 1), state_all(k, 17, t - 1), state_all(k, 16, t)])
            if state_all(k, 16, t - 1) && state_all(k, 16, t)
                Q_table_all(state_all(k, 16, t - 1), state_all(k, 17, t - 1), k) = ...
                    (1 - al_Q) * Q_table_all(state_all(k, 16, t - 1), state_all(k, 17, t - 1), k) + ...
                    al_Q * (state_all(k, 10, t) + gam * max(Q_table_all(state_all(k, 16, t), :, k)));
            elseif state_all(k, 16, t - 1)
                Q_table_all(state_all(k, 16, t - 1), state_all(k, 17, t - 1), k) = ...
                    (1 - al_Q) * Q_table_all(state_all(k, 16, t - 1), state_all(k, 17, t - 1), k) + ...
                    al_Q * (state_all(k, 10, t) + gam * Q_table_all(N_Q + 1, 1, k));
            elseif state_all(k, 16, t)
                Q_table_all(N_Q + 1, 1, k) = ...
                    (1 - al_Q) * Q_table_all(N_Q + 1, 1, k) + ...
                    al_Q * (state_all(k, 10, t) + gam * max(Q_table_all(state_all(k, 16, t), :, k)));
            else
                Q_table_all(N_Q + 1, 1, k) = ...
                    (1 - al_Q) * Q_table_all(N_Q + 1, 1, k) + ...
                    al_Q * (state_all(k, 10, t) + gam * Q_table_all(N_Q + 1, 1, k));
            end
        end

        % 11
        % 17
        [state_all(:, 11, t), state_all(:, 17, t)] = neighbour_to_imitate_Q(state_all(:, 5, t), adj_mat_temp, state_all(:, 10, t), Q_table_all, epsilon_greedy);
        % 12
        state_all(:, 12, t) = state_all(:, 11, t) + (~state_all(:, 11, t)) .* (1:N)';
        % 13
        state_all(:, 13, t) = 1 ./ (1 + exp((state_all(:, 10, t) - state_all(state_all(:, 12, t), 10, t)) ./ be));
        % 14
        state_all(:, 14, t) = rand(N, 1) < state_all(:, 13, t);
        % 15
        state_all(:, 15, t) = state_all(:, 14, t) .*  state_all(:, 12, t) + (~state_all(:, 14, t)) .* (1:N)';        
    
        if ~mod(t, floor(num_to_sim / 11))
            fprintf('第%d组参数, alpha = %3.1f, r = %5.2f, eta = %3.1f, 实验: %d/%d, 进度: %3.1f%% \n', group_ind, al, r, eta, b_mc, B, t / num_to_sim * 100);
        end

    end

    % calculate <C>, <V_a>
    lev = mean(level(burn_in + 1 + (1:mc_step), :), 1);
    lev(3) = 1 - mean(level(burn_in + 1 + (1:mc_step), 2) .^ 4) / mean(level(burn_in + 1 + (1:mc_step), 2) .^ 2) ^ 2 / 3;


%     plot
    if if_plot 
        axis tight manual
        v = VideoWriter('vicsek_demo', "MPEG-4");
        open(v);

        for t = 2:(num_to_sim + 1)
            dx = cos(state_all(:, 3, t));
            dy = sin(state_all(:, 3, t));
        
            index = state_all(:, 4, t) == 1;
            
        %     figure(1)
        %     
            quiver(state_all(index, 1, t), state_all(index, 2, t), dx(index), dy(index), 'b', 'AutoScaleFactor', 0.3);
            hold on
            quiver(state_all(~index, 1, t), state_all(~index, 2, t), dx(~index), dy(~index), 'r', 'AutoScaleFactor', 0.2);
            xlim([0, L]);  
            ylim([0, L]);  
            hold off
        
            frame = getframe(gcf);
            writeVideo(v, frame);
        end          

        close(v);
    end
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





