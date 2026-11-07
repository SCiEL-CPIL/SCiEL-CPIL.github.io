"""Optimal market making with inventory risk - HJB-based policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for precomputed policies
_policy_cache = {}
_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def _build_policy(params):
    """Build optimal bid/ask depth policy table via backward HJB induction."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key in _policy_cache:
        return _policy_cache[key]

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    # Depth grid: include 0 (no quote) and a range of positive depths
    # Max useful depth: where fill prob ~ 1, i.e., A*exp(-k*d)*dt ~ 1 => d ~ ln(A*dt)/k
    # But also need fine resolution near 1/k for the seed behavior
    max_depth = max(3.0 / k, 10.0 / k)
    n_depths = 200
    depths = np.linspace(0.0, max_depth, n_depths)

    # Inventory states: -Q to Q
    q_states = np.arange(-Q, Q + 1)
    n_q = len(q_states)  # 2*Q+1 = 61

    # Precompute fill probabilities for each depth
    # P_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_term = np.exp(-k * depths)
    fill_prob = 1.0 - np.exp(-A * exp_term * dt)  # shape (n_depths,)

    # Value function: V[t, q_index] where q_index maps to inventory q
    # V has shape (steps+1, n_q)
    # V[steps, :] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_q))

    # Policy tables: store optimal bid/ask depth index for each (step, q)
    # bid_policy[step, q_idx] = index into depths for bid
    # ask_policy[step, q_idx] = index into depths for ask
    bid_policy = np.zeros((steps, n_q), dtype=np.int32)
    ask_policy = np.zeros((steps, n_q), dtype=np.int32)

    # Backward induction
    for t in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE:
            # Fallback: use 1/k depth
            for qi in range(n_q):
                bid_policy[t, qi] = 0
                ask_policy[t, qi] = 0
            break

        V_next = V[t + 1]
        V_curr = V[t]

        for qi in range(n_q):
            q = q_states[qi]

            # Inventory cost for this step
            inv_cost = 0.5 * gamma * sigma**2 * q**2 * dt

            best_bid_val = -np.inf
            best_ask_val = -np.inf
            best_bid_idx = 0
            best_ask_idx = 0

            # Try each bid depth
            # If bid fills: inventory becomes q+1 (unless q == Q)
            # If no bid fill: inventory stays q
            # Bid earns its depth if it fills
            if q < Q:
                q_bid_fill = q + 1
                qi_bid_fill = q_bid_fill + Q
                for di in range(n_depths):
                    d = depths[di]
                    pf = fill_prob[di]
                    # Expected value: pf * (d + V_next[qi_bid_fill]) + (1-pf) * V_next[qi]
                    val = pf * (d + V_next[qi_bid_fill]) + (1.0 - pf) * V_next[qi]
                    if val > best_bid_val:
                        best_bid_val = val
                        best_bid_idx = di
            else:
                # q == Q: no bid fill possible, bid depth = 0 (no quote)
                best_bid_val = V_next[qi]
                best_bid_idx = 0  # depth 0 = no quote

            # Try each ask depth
            # If ask fills: inventory becomes q-1 (unless q == -Q)
            # If no ask fill: inventory stays q
            # Ask earns its depth if it fills
            if q > -Q:
                q_ask_fill = q - 1
                qi_ask_fill = q_ask_fill + Q
                for di in range(n_depths):
                    d = depths[di]
                    pf = fill_prob[di]
                    # Expected value: pf * (d + V_next[qi_ask_fill]) + (1-pf) * V_next[qi]
                    val = pf * (d + V_next[qi_ask_fill]) + (1.0 - pf) * V_next[qi]
                    if val > best_ask_val:
                        best_ask_val = val
                        best_ask_idx = di
            else:
                # q == -Q: no ask fill possible
                best_ask_val = V_next[qi]
                best_ask_idx = 0

            # Total expected value for this state
            V_curr[qi] = best_bid_val + best_ask_val - inv_cost

            bid_policy[t, qi] = best_bid_idx
            ask_policy[t, qi] = best_ask_idx

    # Store in cache
    result = {
        'depths': depths,
        'bid_policy': bid_policy,
        'ask_policy': ask_policy,
        'q_states': q_states,
        'n_q': n_q,
    }
    _policy_cache[key] = result
    return result


def quote(t, inventory, mid, params):
    """Return optimal bid/ask depths for each path."""
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple 1/k depth
        depth = np.full(len(inventory), 1.0 / params['k'])
        return depth, depth.copy()

    policy = _build_policy(params)
    depths = policy['depths']
    bid_policy = policy['bid_policy']
    ask_policy = policy['ask_policy']
    Q = params['Q']

    # Determine step index from t
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = max(0, min(step, params['steps'] - 1))

    n_paths = len(inventory)
    bid_depths = np.empty(n_paths, dtype=np.float64)
    ask_depths = np.empty(n_paths, dtype=np.float64)

    for i in range(n_paths):
        q = int(inventory[i])
        qi = q + Q
        if qi < 0 or qi >= policy['n_q']:
            # Out of bounds, use 1/k
            bid_depths[i] = 1.0 / params['k']
            ask_depths[i] = 1.0 / params['k']
            continue

        bid_idx = bid_policy[step, qi]
        ask_idx = ask_policy[step, qi]

        bid_depths[i] = depths[bid_idx]
        ask_depths[i] = depths[ask_idx]

    # Clip to valid range: must be >= -3/k (but our depths are >= 0, so fine)
    # Also ensure no NaN or -inf
    min_valid = -3.0 / params['k']
    bid_depths = np.where(bid_depths >= min_valid, bid_depths, np.inf)
    ask_depths = np.where(ask_depths >= min_valid, ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
