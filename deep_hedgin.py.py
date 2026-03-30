import numpy as np
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

# Black-Scholes pricing

def black_scholes_call(S, K, T, sigma, r=0):
    if T < 10**-6:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def delta(S, K, T, sigma, r=0):
    if T < 10**-6:
        if S > K:
            return 1
        elif S < K:
            return 0.0
        else:
            return 0.5  # Convention
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# Parameters
S0 = 100
K = 100
sigma = 0.01
T = 10
D = 5 
dt = 1/D
N_steps = int(T/dt)
kappa = 0.1

# Q-learning setup
actions = np.arange(-30, 30, 1)  # can trade -30 to +30 shares


def cost(n):
    tick_size = 0.1
    multiplier = 0
    return multiplier * tick_size *( np.abs(n) + 0.01 * n**2)

# Training loop
episodes = 1500
gamma = 0.9   # discount factor
alpha = 1   # learning rate

#BATCH 1 
X = []
Y = []
print("batch : %s"%1)
epsilon = 1  # exploration rate
for ep in range(episodes):
    S = S0
    holdings = 0
    for t in range(N_steps):
        T_remain = T - t*dt
        if np.random.rand() < epsilon:
            action_idx = np.random.choice(len(actions))

        trade = actions[action_idx]

        # Calculate transaction costs and update holdings
        trans_cost = cost(trade)
        nt = holdings     #current holdings
        holdings += trade   #holdings at period ]t+1,t+2] 

        # simulate next price
        S_next = S * np.exp(-0.5*sigma**2*dt + sigma*np.sqrt(dt)*np.random.randn())

        # Compute hedging error and reward
        option_price_diff =100*( black_scholes_call(S_next, K, T_remain-dt, sigma) - black_scholes_call(S, K, T_remain, sigma))
        hedge_pnl = nt * (S_next - S) - trans_cost
        
        delta_w = hedge_pnl + option_price_diff
        reward = delta_w - (kappa/2) * delta_w**2
        
        features = np.array([S, T_remain, nt, trade])
        X.append(features)

        # Update Q
        q_target = reward
        Y.append(reward)

        S = S_next
        
model = ExtraTreesRegressor(n_estimators=50)
model.fit(X, Y)


def predict(S, T_remain, nt):
    x_batch = np.array([[S, T_remain, nt, actions[idx_action]] for idx_action in range(len(actions))])
    q_values = model.predict(x_batch)
    return np.array(q_values)
        
        
        
#BATCH 2 ..5
b=2
epsilon = 0.6
while b <= 5:
    print("batch : %s"%b)
    X = []
    Y = []
    epsilon -= 0.1
    for ep in range(episodes):
        S = S0
        holdings = 0
        for t in range(N_steps):
            if t == 0:
                #current state action vector
                if np.random.rand() < epsilon:
                    action_idx = np.random.choice(len(actions))
                else:
                    action_idx = np.argmax(predict(S, T, 0))

                trade = actions[action_idx]  #at 
                current_state_action = [S, T, 0, trade]    #(st,at)
                
            nt = holdings #current holdings at ]t,t+1] 
            
            # Calculate transaction costs
            trade = current_state_action[-1]
            trans_cost = cost(trade)

            # simulate next price
            S_next = S * np.exp(-0.5*sigma**2*dt + sigma*np.sqrt(dt)*np.random.randn())

            # Compute reward
            T_remain = T - t*dt
            option_price_diff = 100*( black_scholes_call(S_next, K, T_remain-dt, sigma) - black_scholes_call(S, K, T_remain, sigma))
            hedge_pnl = nt * (S_next - S) - trans_cost
            delta_w = hedge_pnl + option_price_diff
            reward = delta_w - (kappa/2) * delta_w**2
            
            #update holdings
            holdings += trade   #nt+1 : holdings at ]t+1,t+2] 
            
            #nest state action vector
            T_remain = T - (t+1)*dt
            if np.random.rand() < epsilon:
                action_idx = np.random.choice(len(actions))
            else:
                action_idx = np.argmax(predict(S_next, T_remain, holdings))

            trade = actions[action_idx]  #at+1
            next_state_action = [S_next, T_remain, holdings, trade]      #(st+1, at+1)


            features = np.array(current_state_action)   #(st,at)
            X.append(features)

            # Update q
            q_target = reward + gamma * model.predict([next_state_action])[0]
            Y.append(q_target)

            S = S_next
            current_state_action = next_state_action
        
    model.fit(X, Y)
    b+=1

# Test the learned policy
S = S0
holdings = 0
total_pnl = 0
trades_cost = 0
deltas = [delta(S, K, T, sigma)]
price_list = [S]
holdings_list = [0]
cost_list = [0]


option_pnl_list = []
stock_pnl_list = []
total_pnl_list = []



for t in range(N_steps):
    T_remain = T - t*dt
    action_idx = np.argmax(predict(S, T_remain, holdings))
    trade = actions[action_idx]

    trans_cost = cost(trade)
    trades_cost += trans_cost
    cost_list.append(trades_cost)
    

    S_next = S * np.exp(-0.5*sigma**2*dt + sigma*np.sqrt(dt)*np.random.randn())
    price_list.append(S_next)
    deltas.append(delta(S_next, K, T_remain, sigma))

    #option pnl
    option_pnl = 100*(black_scholes_call(S_next, K, T_remain-dt, sigma) - black_scholes_call(S, K, T_remain, sigma))
    option_pnl_list.append(option_pnl)
    #stock_pnl
    stock_pnl = holdings * (S_next - S) - trans_cost
    stock_pnl_list.append(stock_pnl)
    #total pnl
    total_pnl_list.append(option_pnl + stock_pnl)
    
    #holdings at t+1
    holdings += trade
    holdings_list.append(holdings)
    

    S = S_next
    
timesteps = np.arange(0, 51)

def single_plot(y, label, linestyle, color='black', lw=2):
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, y, linestyle=linestyle, color=color, linewidth=lw, label=label)
    plt.xlabel('Timestep (D*T)', fontsize=12, fontweight='bold')
    plt.ylabel('Value (dollars or shares)', fontsize=12, fontweight='bold')
    plt.title(label, fontsize=14)
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


single_plot(deltas, 'delta', '-')
single_plot(price_list, 'underlying price', '--')
single_plot(holdings_list, 'stock_pos_shares', '--')
single_plot(cost_list, 'cost ', '-', color='blue')


timesteps = np.arange(0,51)
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(timesteps, -100*np.array(deltas), color = 'black', linestyle='-', label='-100Delta')
ax.plot(timesteps, holdings_list, color = 'red', linestyle='-', label='stock.pos.shares')
ax.set_xlabel('Timestep (D*T)', fontsize=12, fontweight='bold')
ax.set_ylabel('Value (dollars or shares)', fontsize=12, fontweight='bold')
ax.legend(frameon=False)
ax.grid(True, linestyle='--', alpha=0.5)

timesteps = np.arange(1,51)
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(timesteps, np.cumsum(option_pnl_list), color = 'green', linestyle='--', label='option.pnl')
ax.plot(timesteps, np.cumsum(stock_pnl_list), color = 'blue', linestyle='-.', label='stock.pnl')
ax.plot(timesteps, np.cumsum(option_pnl_list) + np.cumsum(stock_pnl_list), color = 'red', linestyle='-', label='o.pnl')
ax.set_xlabel('Timestep (D*T)', fontsize=12, fontweight='bold')
ax.set_ylabel('Value (dollars or shares)', fontsize=12, fontweight='bold')
ax.legend(frameon=False)
ax.grid(True, linestyle='--', alpha=0.5)
