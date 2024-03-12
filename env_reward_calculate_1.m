% Environment Setup: 
% First, lets define the environment, including the state space and the reward matrix.
indices = [];
range = 0:0.1:1;
n = length(range);
% rewards = zeros(n, n, n); % Initialize rewards matrix

% Assuming A_t (average latency) is calculated elsewhere and available
A_t = 15; % Placeholder value
% A_t_imaginary = 10000;

for W_d_index = 1:n
    for W_l_index = 1:n
        for W_ec_index = 1:n
            W_d = range(W_d_index);
            W_l = range(W_l_index);
            W_ec = range(W_ec_index);       
            if W_d + W_l + W_ec == 1
                % Update rewards based on A_t
                rewards(W_d_index, W_l_index, W_ec_index) = 1 / A_t;
                indices = [indices; W_d_index, W_l_index, W_ec_index];
                %else 
                %rewards(W_d_index, W_l_index, W_ec_index) = 1 / A_t_imaginary;
            end
        end
    end
end
% Create a table from the indices array
index_rewards_table = array2table(indices, 'VariableNames', {'W_d_I', 'W_l_I', 'W_ec_I'});

% Initialize Q_values
Q_values = zeros(n, n, n, 12); % Assuming 9 possible actions
epsilon = 0.9; % Exploration rate
gamma = 0.9; % Discount factor
alpha = 0.9; % Learning rate

% Agent Training:Implement the agent training loop, updating Q-values based on actions taken and rewards received.
for episode = 1:100 % Number of episodes
    % Get starting state
    [W_d_index, W_l_index, W_ec_index] = Get_Starting_State(rewards,index_rewards_table);
    disp(W_d_index);
    disp(W_l_index);
    disp(W_ec_index);
    while ~Is_Termination_State(W_d_index, W_l_index, W_ec_index, rewards)
        action = Get_Next_Action(Q_values, [W_d_index, W_l_index, W_ec_index], epsilon);
        [new_W_d_Index, new_W_l_Index, new_W_ec_Index] = Get_Next_State(W_d_index, W_l_index, W_ec_index, action);
        
        % Calculate reward for the new state
        reward = rewards(new_W_d_Index, new_W_l_Index, new_W_ec_Index);
        
        % Update Q-values
        old_Q_value = Q_values(W_d_index, W_l_index, W_ec_index, action);
        TD = reward + gamma * max(Q_values(W_d_index, W_l_index, W_ec_index, :)) - old_Q_value;
        new_Q_value = old_Q_value + alpha * TD;
        
        Q_values(W_d_index, W_l_index, W_ec_index, action) = new_Q_value;
        
        % Update state
        W_d_index = new_W_d_Index;
        W_l_index = new_W_l_Index;
        W_ec_index = new_W_ec_Index;
    end
end


%This function checks if the current state is a terminal state based on the rewards matrix. It returns a boolean value indicating whether the state is terminal.

function isTerminal = Is_Termination_State(W_d_index, W_l_index, W_ec_index, rewards)
    % Assuming MAX_R_t is defined elsewhere as the maximum reward
    %MAX_R_t = max(rewards(:));
    MAX_R_t = 1/5;  %set A_t = 5
    if rewards(W_d_index, W_l_index, W_ec_index) >= MAX_R_t
        isTerminal = true;
    else
        isTerminal = false;
    end
end

%This function generates a random starting state that is not a terminal state. It returns the indices for W_d, W_l, and W_ec within the range matrix.

function [W_d_index, W_l_index, W_ec_index] = Get_Starting_State(rewards,index_rewards_table)
    isTerminal = true;
    
    while isTerminal
        % Randomly select indices for W_d, W_l, and W_ec
        numRows_I_table = size(index_rewards_table, 1);
        randomIndex = randi([1, numRows_I_table]);
        randomRow_I_table = index_rewards_table(randomIndex, :);
        W_d_I = randomRow_I_table(1, 1);
        W_d_index = W_d_I{1,1};
        W_l_I = randomRow_I_table(1, 2);
        W_l_index = W_l_I{1,1};
        W_ec_I = randomRow_I_table(1, 3);
        W_ec_index = W_ec_I{1,1};
        isTerminal = Is_Termination_State(W_d_index, W_l_index, W_ec_index, rewards);
        
        if isTerminal
            [W_d_index, W_l_index, W_ec_index] = Get_Starting_State(rewards,index_rewards_table);
        else
            return;
        end
    end
end


%Action Selection: Define the function to select an action based on the current state and epsilon for exploration.

function action = Get_Next_Action(Q_values, state, epsilon)
    if rand() < epsilon
        % Exploration: choose a random action
        action = randi([1, 12]); % Assuming 12 possible actions
    else
        % Exploitation: choose the best action from Q_values
        [~, action] = max(Q_values(state(1), state(2), state(3), :));
    end
end

%State Transition:Define the function to get the next state based on the current state and action taken.

function [new_W_d_Index, new_W_l_Index, new_W_ec_Index] = Get_Next_State(W_d_index, W_l_index, W_ec_index, action)
    step = 0.1;
    % Action to state transition logic
    switch action
        case 1 % increment both W_d & W_l
            new_W_d_Index = W_d_index + 1;  % it is as equal as assigning new_W_d = W_d + 0.1
            new_W_l_Index = W_l_index + 1;  % it is as equal as assigning new_W_l = W_l + 0.1
            new_W_ec_Index = W_ec_index -2; % it is as equal as assigning new_W_EC = W_EC - 0.2
        case 2 % increment W_d & decrement W_l
            new_W_d_Index = W_d_index + 1;
            new_W_l_Index = W_l_index - 1;
            new_W_ec_Index = W_ec_index;
        case 3 % increment both W_d & W_ec


        case 4 % increment W_d & decrement W_ec


        case 5 % decrement W_d & increment W_l


        case 6 % decrement W_d & decrement W_l


        case 7

        case 8

        case 9 % Stay in W_d & increment W_l
            new_W_d_Index = W_d_index;      % it is as equal as assigning new_W_d = W_d
            new_W_l = W_l;
            new_W_ec = W_ec;
        
        case 10

        case 11

        case 12
        % Add cases for other actions...
        % Increment or decrement W_d, W_l, W_ec based on the action
    end
end




