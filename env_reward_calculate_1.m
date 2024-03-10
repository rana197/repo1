%Environment Setup: 
%First, lets define the environment, including the state space and the reward matrix.

range = 0:0.05:1;
n = length(range);
rewards = zeros(n, n, n); % Initialize rewards matrix

% Assuming A_t (average latency) is calculated elsewhere and available
A_t = 1; % Placeholder value

for W_d_index = 1:n
    for W_l_index = 1:n
        for W_ec_index = 1:n
            W_d = range(W_d_index);
            W_l = range(W_l_index);
            W_ec = range(W_ec_index);
            if W_d + W_l + W_ec == 1
                % Update rewards based on A_t
                rewards(W_d_index, W_l_index, W_ec_index) = 1 / A_t;
            end
        end
    end
end

%This function checks if the current state is a terminal state based on the rewards matrix. It returns a boolean value indicating whether the state is terminal.

function isTerminal = Is_Termination_State(W_d_index, W_l_index, W_ec_index, rewards)
    % Assuming MAX_R_t is defined elsewhere as the maximum reward
    %MAX_R_t = max(rewards(:));
    MAX_R_t = 1/50;  %set A_t = 50
    
    if rewards(W_d_index, W_l_index, W_ec_index) == MAX_R_t
        isTerminal = true;
    else
        isTerminal = false;
    end
end

%This function generates a random starting state that is not a terminal state. It returns the indices for W_d, W_l, and W_ec within the range matrix.

function [W_d_index, W_l_index, W_ec_index] = Get_Starting_State(rewards)
    n = 1;
    isTerminal = true;
    
    while isTerminal
        % Randomly select indices for W_d, W_l, and W_ec
        W_d_index = rand([1, n]);
        W_l_index = rand([1, n]);
        W_ec_index = rand([1, n]);
        
        W_all = W_d_index + W_l_index + W_ec_index;
        
        W_d_index = W_d_index/W_all;
        W_l_index = W_l_index/W_all;
        W_ec_index = W_ec_index/W_all;
        
        isTerminal = Is_Termination_State(W_d_index, W_l_index, W_ec_index, rewards);
        
        if isTerminal
            [W_d_index, W_l_index, W_ec_index] = Get_Starting_State(rewards);
        else
            return;
        end
    end
end



%Action Selection: Define the function to select an action based on the current state and epsilon for exploration.

function action = Get_Next_Action(Q_values, state, epsilon)
    if rand() < epsilon
        % Exploration: choose a random action
        action = randi([1, 9]); % Assuming 9 possible actions
    else
        % Exploitation: choose the best action from Q_values
        [~, action] = max(Q_values(state(1), state(2), state(3), :));
    end
end

%State Transition:Define the function to get the next state based on the current state and action taken.

function [new_W_d, new_W_l, new_W_ec] = Get_Next_State(W_d, W_l, W_ec, action)
    step = 0.05;
    % Action to state transition logic
    switch action
        case 1 % Stay in W_d
            new_W_d = W_d;
            new_W_l = W_l;
            new_W_ec = W_ec;
        % Add cases for other actions...
        % Increment or decrement W_d, W_l, W_ec based on the action
    end
end


% Agent Training:Implement the agent training loop, updating Q-values based on actions taken and rewards received.

% Initialize Q_values
Q_values = zeros(n, n, n, 9); % Assuming 9 possible actions
epsilon = 0.9; % Exploration rate
gamma = 0.9; % Discount factor
alpha = 0.9; % Learning rate

for episode = 1:100 % Number of episodes
    % Get starting state
    [W_d, W_l, W_ec] = Get_Starting_State(rewards);
    
    while ~Is_Termination_State(W_d, W_l, W_ec, rewards)
        action = Get_Next_Action(Q_values, [W_d, W_l, W_ec], epsilon);
        [new_W_d, new_W_l, new_W_ec] = Get_Next_State(W_d, W_l, W_ec, action);
        
        % Calculate reward for the new state
        reward = rewards(new_W_d, new_W_l, new_W_ec);
        
        % Update Q-values
        old_Q_value = Q_values(W_d, W_l, W_ec, action);
        TD = reward + gamma * max(Q_values(new_W_d, new_W_l, new_W_ec, :)) - old_Q_value;
        new_Q_value = old_Q_value + alpha * TD;
        
        Q_values(W_d, W_l, W_ec, action) = new_Q_value;
        
        % Update state
        W_d = new_W_d;
        W_l = new_W_l;
        W_ec = new_W_ec;
    end
end