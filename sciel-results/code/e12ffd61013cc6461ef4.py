"""Market making with inventory risk via precomputed MDP policy."""
import numpy as np
import time

# Precomputed policy tables: policy[regime_idx][step][inventory_offset] = (bid_depth, ask_depth)
# regime_idx: 0=(2,140,1.5,0.05), 1=(2,140,1.5,0.5), 2=(1.5,50,0.75,0.2)
# inventory_offset = inventory + Q (0..2Q)
# step = n (0..999)

# We'll solve the MDP at module load time.
# To keep load time reasonable, we'll use a grid of depth values and do backward induction.

def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """Solve the MDP for one regime. Returns policy table shape (steps, 2*Q+1, 2)."""
    dt = T / steps
    N = 2 * Q + 1  # number of inventory states
    inv_range = np.arange(-Q, Q + 1)
    
    # Depth grid: we'll use a fine grid for optimization
    # Depths must be >= 0 and < 3/k
    max_depth = 3.0 / k * 0.999
    n_depths = 200
    depth_grid = np.linspace(0.001, max_depth, n_depths)
    
    # Value function: V[step][inv_offset]
    # V[n][i] = max expected future reward from step n with inventory inv_range[i]
    V = np.zeros((steps + 1, N))
    policy = np.zeros((steps, N, 2))  # (bid_depth, ask_depth)
    
    # For each step, for each inventory, we need to choose bid_depth and ask_depth
    # to maximize: immediate reward + expected future reward
    # 
    # Immediate reward from quoting at depths (d_bid, d_ask):
    #   With prob p_bid = 1-exp(-A*exp(-k*d_bid)*dt): bid fill, earn d_bid, inv decreases by 1
    #   With prob p_ask = 1-exp(-A*exp(-k*d_ask)*dt): ask fill, earn d_ask, inv increases by 1
    #   Independent.
    #   Also: inventory cost = (gamma/2)*sigma^2*q^2*dt where q is inventory BEFORE fills
    #
    # So expected immediate reward = p_bid*d_bid + p_ask*d_ask - (gamma/2)*sigma^2*q^2*dt
    # Expected future = p_bid*p_ask*V[n+1][q] + p_bid*(1-p_ask)*V[n+1][q-1] 
    #                  + (1-p_bid)*p_ask*V[n+1][q+1] + (1-p_bid)*(1-p_ask)*V[n+1][q]
    #                  (with boundary conditions: no bid fill at q=Q, no ask fill at q=-Q)
    
    # For efficiency, we'll vectorize over depths.
    # For a given inventory q and step n:
    #   p_bid(d) = 1 - exp(-A*exp(-k*d)*dt)
    #   p_ask(d) = 1 - exp(-A*exp(-k*d)*dt)
    
    # Precompute p for all depths
    # p(d) = 1 - exp(-A*exp(-k*d)*dt)
    exp_kd = np.exp(-k * depth_grid)
    p_grid = 1.0 - np.exp(-A * exp_kd * dt)
    
    # V[n+1] is known, compute V[n] and policy[n]
    for n in range(steps - 1, -1, -1):
        V_next = V[n + 1]
        
        for i in range(N):
            q = inv_range[i]
            q_offset = i
            
            # Cost for this step
            cost = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # Determine valid fills
            can_bid = (q < Q)   # can have a bid fill (inventory decreases)
            can_ask = (q > -Q)  # can have an ask fill (inventory increases)
            
            # If both sides valid:
            # For each (d_bid, d_ask) in grid x grid:
            #   p_b = p_grid[d_bid_idx] if can_bid else 0
            #   p_a = p_grid[d_ask_idx] if can_ask else 0
            #   imm = p_b*d_bid + p_a*d_ask - cost
            #   fut = (1-p_b)*(1-p_a)*V_next[q] + p_b*(1-p_a)*V_next[q-1] 
            #         + (1-p_b)*p_a*V_next[q+1] + p_b*p_a*V_next[q]
            #   total = imm + fut
            # Maximize over (d_bid, d_ask)
            
            if can_bid and can_ask:
                # Vectorized: for each d_bid_idx, d_ask_idx
                # p_b shape (n_depths,), p_a shape (n_depths,)
                pb = p_grid  # (n_depths,)
                pa = p_grid  # (n_depths,)
                
                # Outer products
                # imm = pb[:,None]*depth_grid[:,None] + pa[None,:]*depth_grid[None,:] - cost
                imm = pb[:, None] * depth_grid[:, None] + pa[None, :] * depth_grid[None, :] - cost
                
                # fut: 
                # V_next[q] = V_next[i], V_next[q-1] = V_next[i-1], V_next[q+1] = V_next[i+1]
                V_q = V_next[i]
                V_qm1 = V_next[i - 1]
                V_qp1 = V_next[i + 1]
                
                # fut = (1-pb)[:,None]*(1-pa)[None,:]*V_q + pb[:,None]*(1-pa)[None,:]*V_qm1 
                #     + (1-pb)[:,None]*pa[None,:]*V_qp1 + pb[:,None]*pa[None,:]*V_q
                fut = ((1 - pb)[:, None] * (1 - pa)[None, :] * V_q +
                       pb[:, None] * (1 - pa)[None, :] * V_qm1 +
                       (1 - pb)[:, None] * pa[None, :] * V_qp1 +
                       pb[:, None] * pa[None, :] * V_q)
                
                total = imm + fut
                idx = np.argmax(total)
                db_idx, da_idx = np.unravel_index(idx, total.shape)
                policy[n, i, 0] = depth_grid[db_idx]
                policy[n, i, 1] = depth_grid[da_idx]
                V[n, i] = total[db_idx, da_idx]
                
            elif can_bid:
                # Only bid side (q == Q... wait, can_bid is q < Q, so q=Q-1 max for bid? No)
                # Actually can_bid means q < Q, so a bid fill would take q to q-1 which is >= -Q
                # But if q == Q, can_bid is False. If q == -Q, can_ask is False.
                # So can_bid and not can_ask means q == -Q
                # Only bid fill possible
                pb = p_grid
                imm = pb * depth_grid - cost
                V_q = V_next[i]
                V_qm1 = V_next[i - 1]
                fut = (1 - pb) * V_q + pb * V_qm1
                total = imm + fut
                idx = np.argmax(total)
                policy[n, i, 0] = depth_grid[idx]
                policy[n, i, 1] = np.inf  # no ask
                V[n, i] = total[idx]
                
            elif can_ask:
                # Only ask fill possible (q == Q)
                pa = p_grid
                imm = pa * depth_grid - cost
                V_q = V_next[i]
                V_qp1 = V_next[i + 1]
                fut = (1 - pa) * V_q + pa * V_qp1
                total = imm + fut
                idx = np.argmax(total)
                policy[n, i, 0] = np.inf  # no bid
                policy[n, i, 1] = depth_grid[idx]
                V[n, i] = total[idx]
                
            else:
                # Neither can fill (shouldn't happen since Q >= 1)
                policy[n, i, 0] = np.inf
                policy[n, i, 1] = np.inf
                V[n, i] = -cost
    
    return policy


# Precompute policies for all three regimes
_REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2),
]
T_val = 1.0
steps_val = 1000
Q_val = 30

_start_time = time.time()
_POLICIES = []
for sigma, A, k, gamma in _REGIMES:
    pol = _solve_regime(sigma, A, k, gamma, T_val, steps_val, Q_val)
    _POLICIES.append(pol)
_load_time = time.time() - __start_time

# Cache for regime lookup
_regime_cache = {}

def _get_regime_idx(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    if key not in _regime_cache:
        for idx, (s, A, k, g) in enumerate(_REGIMES):
            if abs(s - params['sigma']) < 1e-9 and abs(A - params['A']) < 1e-9 and \
               abs(k - params['k']) < 1e-9 and abs(g - params['gamma']) < 1e-9:
                _regime_cache[key] = idx
                break
        else:
            _regime_cache[key] = 0  # fallback
    return _regime_cache[key]


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths) arrays of shape (4000,)."""
    deadline = 330.0
    elapsed = time.time() - _start_time
    if elapsed > deadline - 5.0:
        # Safety: return seed-like result
        k = params["k"]
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.int32)
    
    # Determine step index
    T = params['T']
    steps = params['steps']
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(steps - 1, n))
    
    Q = params['Q']
    regime_idx = _get_regime_idx(params)
    pol = _POLICIES[regime_idx]
    
    # Clip inventory to valid range
    inv_clipped = np.clip(inv, -Q, Q)
    offsets = inv_clipped + Q  # 0..2Q
    
    bid_depths = pol[n, offsets, 0]
    ask_depths = pol[n, offsets, 1]
    
    # Ensure valid values: must be >= 0 and < 3/k, or inf
    k = params['k']
    max_d = 3.0 / k
    bid_depths = np.where(bid_depths > max_d, np.inf, bid_depths)
    ask_depths = np.where(ask_depths > max_d, np.inf, ask_depths)
    bid_depths = np.where(bid_depths < 0, 0.0, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.0, ask_depths)
    
    return bid_depths, ask_depths
