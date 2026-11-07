"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = None
_DEADLINE = 320.0  # seconds of wall clock budget for the whole run.

# Cache for pre-computed policy tables per parameter regime
_POLICY_CACHE = {}

def _solve_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the discrete DP for the optimal policy using a quadratic value function approximation.
    Returns a dictionary with arrays for bid and ask depths for each inventory level and time step.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    # If we are running low on time, fall back to a heuristic
    if time.time() - _start_time > _DEADLINE - 60.0:
        return None

    dt = T / steps
    inv_k = 1.0 / k
    Q_int = int(Q)
    n_q = 2 * Q_int + 1
    qs = np.arange(-Q_int, Q_int + 1, dtype=np.float64)
    
    # We will store the optimal bid and ask depths for each time step and inventory level
    # bid_depths[n][q_idx] is the optimal bid depth at time n (0=last step, steps=first step) for inventory qs[q_idx]
    # Actually, let's store it indexed by time step n (0..steps-1) and inventory q.
    # To save memory, we only need the previous step's value function to compute the current step's optimal action.
    # However, the optimal action depends on the *current* value function approximation coefficients, which depend on the *next* step's value function.
    # So we can compute the optimal action for step n using the coefficients of V_{n+1}.
    
    # Let's store the optimal actions for all steps and all inventory levels.
    # bid_opt[n, q_idx] and ask_opt[n, q_idx]
    # n goes from 0 (last step, t=T-dt) to steps-1 (first step, t=0)
    # Wait, the loop in quote is n=0..999. t = n*dt.
    # So n=0 is the first step.
    # Let's index the tables by n (0..steps-1).
    
    bid_opt = np.zeros((steps, n_q))
    ask_opt = np.zeros((steps, n_q))
    
    # Quadratic coefficients for V_n(q) = a_n * q^2 + b_n * q + c_n
    # At the terminal step (n=steps), V = 0.
    a = 0.0
    b = 0.0
    c = 0.0
    
    # We iterate backwards from n = steps-1 down to 0.
    # V_n(q) = max_{bid, ask} { bid + ask - (gamma/2) sigma^2 q^2 dt + E[V_{n+1}(q')] }
    # where q' is the inventory after fills.
    # q' = q + 1 with prob p_bid, q - 1 with prob p_ask, q with prob 1 - p_bid - p_ask.
    # p_bid = 1 - exp(-A exp(-k bid) dt)
    # p_ask = 1 - exp(-A exp(-k ask) dt)
    
    # For a given q, we want to maximize:
    # f(bid, ask) = bid + ask - (gamma/2) sigma^2 q^2 dt + p_bid * V_{n+1}(q+1) + p_ask * V_{n+1}(q-1) + (1-p_bid-p_ask) * V_{n+1}(q)
    
    # Using the quadratic ansatz V_{n+1}(q) = a(q^2) + b(q) + c:
    # V_{n+1}(q+1) = a(q+1)^2 + b(q+1) + c = a q^2 + (2a+b)q + (a+b+c)
    # V_{n+1}(q-1) = a(q-1)^2 + b(q-1) + c = a q^2 + (-2a+b)q + (a-b+c)
    # V_{n+1}(q) = a q^2 + b q + c
    
    # Let V_q = V_{n+1}(q), V_qp = V_{n+1}(q+1), V_qm = V_{n+1}(q-1).
    # f(bid, ask) = bid + ask - C_q + p_bid * V_qp + p_ask * V_qm + (1-p_bid-p_ask) * V_q
    #              = bid + ask - C_q + V_q + p_bid * (V_qp - V_q) + p_ask * (V_qm - V_q)
    
    # Let delta_plus = V_qp - V_q = 2a q + a + b
    # Let delta_minus = V_qm - V_q = -2a q + a - b
    
    # f(bid, ask) = bid + ask - C_q + V_q + p_bid * delta_plus + p_ask * delta_minus
    
    # To find the optimal bid and ask, we can solve the first-order conditions.
    # However, bid and ask appear in p_bid and p_ask in a non-linear way.
    # Let's use the Lambert W approximation or a simple grid search for the optimal bid and ask for each q.
    # Given the constraints, a small grid search for bid and ask for each q might be feasible.
    # Or we can use the fact that the optimal bid and ask are likely to be close to 1/k.
    
    # Let's try a simple grid search for bid and ask for each q.
    # The range for bid and ask is [0, 10/k].
    # Let's use 50 points in this range.
    num_depths = 50
    depths = np.linspace(0.0, 10.0 * inv_k, num_depths)
    
    for n in range(steps - 1, -1, -1):
        # For each q, find the optimal bid and ask
        for q_idx in range(n_q):
            q = qs[q_idx]
            
            # Handle boundaries
            if q >= Q_int:
                # No bid fill possible
                p_bid = 0.0
                # Optimize ask only
                # f(ask) = ask + p_ask * delta_minus
                # Let's just find the best ask from the grid
                best_ask = depths[0]
                best_val = -np.inf
                for ask in depths:
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * ask) * dt)
                    val = ask + p_ask * delta_minus
                    if val > best_val:
                        best_val = val
                        best_ask = ask
                bid_opt[n, q_idx] = np.inf  # No quote
                ask_opt[n, q_idx] = best_ask
            elif q <= -Q_int:
                # No ask fill possible
                p_ask = 0.0
                # Optimize bid only
                best_bid = depths[0]
                best_val = -np.inf
                for bid in depths:
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * bid) * dt)
                    val = bid + p_bid * delta_plus
                    if val > best_val:
                        best_val = val
                        best_bid = bid
                bid_opt[n, q_idx] = best_bid
                ask_opt[n, q_idx] = np.inf  # No quote
            else:
                # Interior q
                # Grid search for bid and ask
                best_bid = depths[0]
                best_ask = depths[0]
                best_val = -np.inf
                for bid in depths:
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * bid) * dt)
                    for ask in depths:
                        p_ask = 1.0 - np.exp(-A * np.exp(-k * ask) * dt)
                        val = bid + ask + p_bid * delta_plus + p_ask * delta_minus
                        if val > best_val:
                            best_val = val
                            best_bid = bid
                            best_ask = ask
                bid_opt[n, q_idx] = best_bid
                ask_opt[n, q_idx] = best_ask
        
        # Update the quadratic coefficients a, b, c for V_n
        # This is the tricky part. We need to approximate V_n(q) by a q^2 + b q + c.
        # We can use least squares to fit the quadratic to the computed V_n(q) values.
        # V_n(q) = max_{bid, ask} f(bid, ask)
        # We have computed the optimal bid and ask for each q. Let's compute V_n(q) for each q.
        
        V_n = np.zeros(n_q)
        for q_idx in range(n_q):
            q = qs[q_idx]
            bid = bid_opt[n, q_idx]
            ask = ask_opt[n, q_idx]
            
            if np.isinf(bid):
                p_bid = 0.0
            else:
                p_bid = 1.0 - np.exp(-A * np.exp(-k * bid) * dt)
            
            if np.isinf(ask):
                p_ask = 0.0
            else:
                p_ask = 1.0 - np.exp(-A * np.exp(-k * ask) * dt)
            
            C_q = 0.5 * gamma * sigma * sigma * q * q * dt
            V_n[q_idx] = bid + ask - C_q + (a * q * q + b * q + c) + p_bid * delta_plus + p_ask * delta_minus
        
        # Fit quadratic V_n(q) ~ a_new q^2 + b_new q + c_new
        # Use np.polyfit
        if n_q >= 3:
            coeffs = np.polyfit(qs, V_n, 2)
            a, b, c = coeffs[0], coeffs[1], coeffs[2]
        
        # Check time
        if time.time() - _start_time > _DEADLINE - 30.0:
            return None

    _POLICY_CACHE[key] = {
        'bid_opt': bid_opt,
        'ask_opt': ask_opt,
        'qs': qs
    }
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        k = params['k']
        base = 1.0 / k
        return np.full(len(inventory), base), np.full(len(inventory), base)
    
    N = len(inventory)
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Get the pre-computed policy
    policy = _solve_policy(sigma, A, k, gamma, T, steps, Q)
    
    if policy is None:
        # Fallback to heuristic
        t_rem = T - t
        inv_float = inventory.astype(np.float64)
        x = gamma * sigma * sigma * t_rem * inv_float
        _BASE_Y = 1.0
        _BASE_EXP = np.exp(_BASE_Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            denom_bid = 1.0 - x
            denom_ask = 1.0 + x
            raw_bid = np.log(_BASE_EXP / denom_bid) / k
            raw_ask = np.log(_BASE_EXP / denom_ask) / k
        bid_depths = np.where(denom_bid <= 0, np.inf, raw_bid)
        ask_depths = np.where(denom_ask <= 0, np.inf, raw_ask)
        min_depth = -3.0 / k + 1e-9
        max_depth = 100.0 / k
        finite_bid = np.isfinite(bid_depths)
        finite_ask = np.isfinite(ask_depths)
        if np.any(finite_bid):
            bid_depths[finite_bid] = np.clip(bid_depths[finite_bid], min_depth, max_depth)
        if np.any(finite_ask):
            ask_depths[finite_ask] = np.clip(ask_depths[finite_ask], min_depth, max_depth)
        at_max_inventory = (inventory == Q)
        at_min_inventory = (inventory == -Q)
        bid_depths[at_max_inventory] = np.inf
        ask_depths[at_min_inventory] = np.inf
        return bid_depths, ask_depths

    bid_opt = policy['bid_opt']
    ask_opt = policy['ask_opt']
    qs = policy['qs']
    Q_int = int(Q)
    n_q = len(qs)
    
    # Determine the current time step index n
    # t = n * dt
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Map inventory to q_idx
    # qs is from -Q to Q with step 1
    q_idx = inventory.astype(int) + Q_int
    # Clamp q_idx to valid range
    q_idx = np.clip(q_idx, 0, n_q - 1)
    
    bid_depths = bid_opt[n, q_idx]
    ask_depths = ask_opt[n, q_idx]
    
    # Ensure no invalid values
    min_depth = -3.0 / k + 1e-9
    bad_bid = np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths <= -3.0/k)
    bad_ask = np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths <= -3.0/k)
    if np.any(bad_bid):
        bid_depths[bad_bid] = min_depth
    if np.any(bad_ask):
        ask_depths[bad_ask] = min_depth
        
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
