"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 290.0  # Safety margin before 330s hard kill
_CACHE = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the optimal depth policy via fully vectorized dynamic programming.
    Computes exact optimal bid/ask depths for all inventory states at all timesteps.
    
    Returns (bid_table, ask_table) where:
      bid_table[t, i] = optimal bid depth at time t, inventory index i
      ask_table[t, i] = optimal ask depth at time t, inventory index i
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _CACHE:
        return _CACHE[key]
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Value function: V[t, i] where i indexes q from -Q to Q
    V = np.zeros((N + 1, n_q))
    # Optimal depths tables
    bid_table = np.zeros((N, n_q))
    ask_table = np.zeros((N, n_q))
    
    # Fine depth grid for searching optimal depth
    # Max useful depth: where probability becomes negligible
    # P(d) = 1 - exp(-A * exp(-k*d) * dt)
    # For large d, P -> 0. We want to cover the range where trade-offs matter.
    # A/k * dt is the characteristic scale. Let's use a grid up to 8.0/k to be safe.
    max_depth = 8.0 / k
    grid_size = 200
    depths = np.linspace(0.0, max_depth, grid_size)
    
    # Precompute probability grids for all depths
    # P[d] = 1 - exp(-A * exp(-k*d) * dt)
    exp_term = np.exp(-k * depths)
    P = 1.0 - np.exp(-A * exp_term * dt)
    
    # Precompute cost per inventory level: (gamma/2) * sigma^2 * q^2 * dt
    cost_per_q = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
    
    # Index offsets for next inventory states
    # q -> q+1 (bid fill): index i -> i+1
    # q -> q-1 (ask fill): index i -> i-1
    # q -> q (both or none): index i -> i
    
    for t in range(N - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE:
            # Fallback: use current V and set remaining to safe defaults
            # Fill remaining timesteps with a simple heuristic
            for tt in range(t, -1, -1):
                if tt < N:
                    # Use symmetric depth based on inventory
                    base_d = 1.0 / k
                    bid_table[tt] = np.clip(base_d + 0.01 * q_vals, 0.0, None)
                    ask_table[tt] = np.clip(base_d - 0.01 * q_vals, 0.0, None)
                    V[tt] = V[tt + 1] if tt + 1 <= N else 0.0
            break
        
        # For each inventory state i, we need to find optimal (d_b, d_a)
        # Vectorize over all i simultaneously.
        # 
        # For state i (inventory q_i):
        #   V_b = V[t+1, i+1] if i+1 < n_q else V[t+1, n_q-1] (but q=Q has no bid fill)
        #   V_a = V[t+1, i-1] if i-1 >= 0 else V[t+1, 0] (but q=-Q has no ask fill)
        #   V_same = V[t+1, i]
        #
        # Actually, the problem states: "no bid fill at q = Q, no ask fill at q = -Q"
        # So at q=Q, bid fill prob is 0. At q=-Q, ask fill prob is 0.
        # We handle this by masking.
        
        # Get next state values for all i
        V_next = V[t + 1, :]  # shape (n_q,)
        
        # V after bid fill (q -> q+1): for i from 0 to n_q-2, next is i+1
        # For i = n_q-1 (q=Q), no bid fill, so this term doesn't matter
        V_bid_next = np.zeros(n_q)
        V_bid_next[:n_q-1] = V_next[1:n_q]
        V_bid_next[n_q-1] = V_next[n_q-1]  # placeholder, will be masked
        
        # V after ask fill (q -> q-1): for i from 1 to n_q-1, next is i-1
        # For i = 0 (q=-Q), no ask fill
        V_ask_next = np.zeros(n_q)
        V_ask_next[1:] = V_next[:n_q-1]
        V_ask_next[0] = V_next[0]  # placeholder, will be masked
        
        V_same_next = V_next.copy()
        
        # Cost for each i
        costs = cost_per_q  # shape (n_q,)
        
        # Now, for each i, we search over (d_b, d_a) in depths x depths
        # To vectorize over i, we create arrays of shape (n_q, grid_size, grid_size)
        # This might be memory intensive: 61 * 200 * 200 * 8 bytes = ~19.5 MB per array
        # We have several such arrays, so ~100-200 MB. Should be OK.
        
        # Pb and Pa for all (i, d_b) and (i, d_a)
        # Since P depends only on depth, not on i:
        # Pb[d_b_idx] is the same for all i, but we need to mask for q=Q
        # Pa[d_a_idx] is the same for all i, but we need to mask for q=-Q
        
        # Let's create meshgrid for depths
        db_grid = depths[None, :, None]  # shape (1, grid_size, 1)
        da_grid = depths[None, None, :]  # shape (1, 1, grid_size)
        
        # Pb shape: (1, grid_size, 1)
        Pb = P[None, :, None]
        # Pa shape: (1, 1, grid_size)
        Pa = P[None, None, :]
        
        # Now broadcast to (n_q, grid_size, grid_size)
        # But we need to apply masks for boundary conditions
        
        # Mask for bid fill: valid if q < Q, i.e., i < n_q-1
        bid_mask = np.ones(n_q, dtype=bool)
        bid_mask[n_q-1] = False  # q=Q, no bid fill
        
        # Mask for ask fill: valid if q > -Q, i.e., i > 0
        ask_mask = np.ones(n_q, dtype=bool)
        ask_mask[0] = False  # q=-Q, no ask fill
        
        # For boundary states, set the respective probability to 0
        # Pb_effective[i, db_idx, da_idx] = Pb[db_idx] if bid_mask[i] else 0
        # Pa_effective[i, db_idx, da_idx] = Pa[da_idx] if ask_mask[i] else 0
        
        # To avoid creating huge masked arrays, we can compute EV in a way that handles masks
        # EV[i, db_idx, da_idx] = 
        #   Pb_eff * (1-Pa_eff) * V_bid_next[i] +
        #   Pa_eff * (1-Pb_eff) * V_ask_next[i] +
        #   Pb_eff * Pa_eff * V_same_next[i] +
        #   (1-Pb_eff) * (1-Pa_eff) * V_same_next[i] +
        #   Pb_eff * db_grid[db_idx] + Pa_eff * da_grid[da_idx] - costs[i]
        
        # Let's simplify the value terms:
        # V_term = V_bid_next[i] * Pb_eff * (1-Pa_eff) + V_ask_next[i] * Pa_eff * (1-Pb_eff) 
        #          + V_same_next[i] * (Pb_eff * Pa_eff + (1-Pb_eff) * (1-Pa_eff))
        # Note: Pb_eff * Pa_eff + (1-Pb_eff) * (1-Pa_eff) = 1 - Pb_eff - Pa_eff + 2*Pb_eff*Pa_eff
        #      = 1 - (Pb_eff + Pa_eff - 2*Pb_eff*Pa_eff)
        #      = 1 - (Pb_eff + Pa_eff - Pb_eff*Pa_eff - Pb_eff*Pa_eff)
        #      This is just the probability of (both fill) + (no fill)
        
        # Actually, it's simpler to just compute it directly.
        # Let's create the full 3D arrays.
        
        # Pb_full: shape (n_q, grid_size, grid_size)
        # We can do this efficiently:
        Pb_full = np.broadcast_to(Pb, (n_q, grid_size, grid_size)).copy()
        Pa_full = np.broadcast_to(Pa, (n_q, grid_size, grid_size)).copy()
        
        # Apply masks
        Pb_full[~bid_mask, :, :] = 0.0
        Pa_full[~ask_mask, :, :] = 0.0
        
        # V_bid_next_full: shape (n_q, 1, 1) broadcast to (n_q, grid_size, grid_size)
        V_bid_next_full = np.broadcast_to(V_bid_next[:, None, None], (n_q, grid_size, grid_size))
        V_ask_next_full = np.broadcast_to(V_ask_next[:, None, None], (n_q, grid_size, grid_size))
        V_same_next_full = np.broadcast_to(V_same_next[:, None, None], (n_q, grid_size, grid_size))
        
        # db_full: shape (1, grid_size, 1) broadcast to (n_q, grid_size, grid_size)
        db_full = np.broadcast_to(db_grid, (n_q, grid_size, grid_size))
        da_full = np.broadcast_to(da_grid, (n_q, grid_size, grid_size))
        
        # costs_full: shape (n_q, 1, 1) broadcast to (n_q, grid_size, grid_size)
        costs_full = np.broadcast_to(costs[:, None, None], (n_q, grid_size, grid_size))
        
        # Compute EV
        EV = (Pb_full * (1 - Pa_full) * V_bid_next_full +
              Pa_full * (1 - Pb_full) * V_ask_next_full +
              Pb_full * Pa_full * V_same_next_full +
              (1 - Pb_full) * (1 - Pa_full) * V_same_next_full +
              Pb_full * db_full +
              Pa_full * da_full -
              costs_full)
        
        # Find optimal (db_idx, da_idx) for each i
        # Flatten the last two dimensions
        EV_flat = EV.reshape(n_q, -1)
        best_flat_idx = np.argmax(EV_flat, axis=1)
        best_db_idx = best_flat_idx // grid_size
        best_da_idx = best_flat_idx % grid_size
        
        # Store optimal depths
        bid_table[t, :] = depths[best_db_idx]
        ask_table[t, :] = depths[best_da_idx]
        
        # Update V[t, i] = max EV for each i
        V[t, :] = EV_flat[np.arange(n_q), best_flat_idx]
    
    _CACHE[key] = (bid_table, ask_table)
    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) arrays of shape (4000,).
    Uses exact precomputed DP tables for the current regime.
    """
    if time.time() > _DEADLINE:
        # Fallback to seed
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    bid_table, ask_table = _solve_regime(
        params["sigma"], params["A"], params["k"], 
        params["gamma"], params["T"], params["steps"], params["Q"]
    )
    
    # t is n * dt, so step index n = int(round(t / dt))
    dt = params["T"] / params["steps"]
    n = int(round(t / dt))
    n = max(0, min(n, params["steps"] - 1))
    
    # inventory is int array in [-Q, Q]
    # Index into table: i = q + Q
    Q = params["Q"]
    idx = inventory + Q
    
    # Clip indices to valid range (should already be valid)
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = bid_table[n, idx]
    ask_depths = ask_table[n, idx]
    
    # Ensure non-negative depths and no NaN/inf
    bid_depths = np.clip(bid_depths, 0.0, None)
    ask_depths = np.clip(ask_depths, 0.0, None)
    
    bid_depths = np.nan_to_num(bid_depths, nan=0.0, posinf=1e6, neginf=0.0)
    ask_depths = np.nan_to_num(ask_depths, nan=0.0, posinf=1e6, neginf=0.0)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
