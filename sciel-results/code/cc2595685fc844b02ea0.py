"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via DP over (time, inventory) states.
    Vectorized over inventory for speed.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    # Depth grid: use finer grid for better accuracy
    depth_min = 1e-4
    depth_max = max(5.0 / k, 0.5)
    n_depths = 60
    depth_grid = np.linspace(depth_min, depth_max, n_depths)

    # V[q] = value function for current time step, indexed by q + Q
    V_next = np.zeros(2 * Q + 1)

    # Store optimal depths: opt_bid[n, q+Q], opt_ask[n, q+Q]
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))

    # Precompute fill probabilities for each depth
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

    # Precompute penalty for each inventory level
    q_vals = np.arange(-Q, Q + 1)
    penalties = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 5.0:
            break

        V_curr = np.zeros(2 * Q + 1)

        # Interior states: can quote both sides
        # Vectorize over inventory for interior states
        for q_idx in range(1, 2 * Q):
            q = q_idx - Q
            penalty = penalties[q_idx]

            # Compute expected next values for each bid depth
            # If bid fills: q+1, if ask fills: q-1, both: q, none: q
            # For each pair (i, j): val = db*p_bid + da*p_ask - penalty + 
            #   p_both*V_next[q] + p_bid_only*V_next[q+1] + p_ask_only*V_next[q-1] + p_none*V_next[q]

            # Use numpy to vectorize over depth pairs
            # Create meshgrid of fill probs
            pb, pa = np.meshgrid(fill_probs, fill_probs, indexing='ij')
            db, da = np.meshgrid(depth_grid, depth_grid, indexing='ij')

            p_both = pb * pa
            p_bid_only = pb * (1 - pa)
            p_ask_only = (1 - pb) * pa
            p_none = (1 - pb) * (1 - pa)

            reward = db * pb + da * pa - penalty
            next_val = (p_both * V_next[q_idx] +
                        p_bid_only * V_next[q_idx + 1] +
                        p_ask_only * V_next[q_idx - 1] +
                        p_none * V_next[q_idx])

            val = reward + next_val
            best_idx = np.unravel_index(np.argmax(val), val.shape)
            best_val = val[best_idx]
            best_bid = depth_grid[best_idx[0]]
            best_ask = depth_grid[best_idx[1]]

            V_curr[q_idx] = best_val
            opt_bid[n, q_idx] = best_bid
            opt_ask[n, q_idx] = best_ask

        # Boundary states
        # q = Q (index 2*Q): no bid quote, only ask
        q_idx = 2 * Q
        q = Q
        penalty = penalties[q_idx]
        best_val = -penalty
        best_ask = 0.0
        for i, d in enumerate(depth_grid):
            p_ask = fill_probs[i]
            val = d * p_ask - penalty + (1 - p_ask) * V_next[q_idx] + p_ask * V_next[q_idx - 1]
            if val > best_val:
                best_val = val
                best_ask = d
        V_curr[q_idx] = best_val
        opt_bid[n, q_idx] = 0.0
        opt_ask[n, q_idx] = best_ask

        # q = -Q (index 0): no ask quote, only bid
        q_idx = 0
        q = -Q
        penalty = penalties[q_idx]
        best_val = -penalty
        best_bid = 0.0
        for i, d in enumerate(depth_grid):
            p_bid = fill_probs[i]
            val = d * p_bid - penalty + (1 - p_bid) * V_next[q_idx] + p_bid * V_next[q_idx + 1]
            if val > best_val:
                best_val = val
                best_bid = d
        V_curr[q_idx] = best_val
        opt_bid[n, q_idx] = best_bid
        opt_ask[n, q_idx] = 0.0

        V_next = V_curr

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with inventory-based skew."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    n = int(round(t / (T / steps)))
    if n >= steps:
        n = steps - 1

    key = (sigma, A, k, gamma, T, steps, Q)

    if key not in _POLICY_CACHE:
        # Try to compute if we have time
        if time.time() < _DEADLINE - 10.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except:
                pass

    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        # Map inventory to index
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx]
        ask_depths = opt_ask[n, q_idx]
        # Ensure valid depths
        bid_depths = np.where(bid_depths < 1e-4, np.inf, bid_depths)
        ask_depths = np.where(ask_depths < 1e-4, np.inf, ask_depths)
        return bid_depths, ask_depths
    else:
        # Fallback: simple inventory-based skew
        base_depth = 1.0 / k
        skew = gamma * sigma**2 * (T - t) / (2.0 * k)
        bid_depths = base_depth + skew * inventory
        ask_depths = base_depth - skew * inventory

        bid_depths = np.clip(bid_depths, 1e-4, 10.0)
        ask_depths = np.clip(ask_depths, 1e-4, 10.0)

        return bid_depths, ask_depths
# EVOLVE-BLOCK-END
