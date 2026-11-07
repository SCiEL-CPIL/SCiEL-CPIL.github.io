"""Optimal market making with inventory risk: precomputed HJB backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

# Cache for precomputed tables
# Key: tuple(params)
# Value: dict with 'bid' and 'ask' tables of shape (steps, 2*Q+1)
_HJB_CACHE = {}


def _solve_hjb(sigma, A, k, gamma, T, steps, Q, deadline):
    """
    Solve the HJB equation for the optimal bid and ask depths.
    Returns tables bid_table and ask_table of shape (steps, 2*Q+1).
    """
    dt = T / steps
    # Grid of candidate depths
    # Max depth: if depth > 5/k, prob is exp(-5) ~ 0.006, reward ~ 5/k * 0.006.
    # 1/k is the risk-neutral peak. Let's go up to 5/k.
    max_depth = 5.0 / k
    # Use a fine grid for accuracy
    num_grid = 200
    grid = np.linspace(0.0, max_depth, num_grid)
    
    # Precompute probabilities for the grid
    # p = 1 - exp(-A * exp(-k * b) * dt)
    exp_k_grid = np.exp(-k * grid)
    p_grid = 1.0 - np.exp(-A * exp_k_grid * dt)
    
    # Tables for optimal depths
    # Indexing: table[n, q + Q]
    bid_table = np.zeros((steps, 2 * Q + 1))
    ask_table = np.zeros((steps, 2 * Q + 1))
    
    # V[n, q]
    # We only need V[n+1] to compute V[n]
    V_next = np.zeros(2 * Q + 1)
    
    # Penalty coefficient
    penalty_coeff = (gamma / 2.0) * (sigma ** 2) * dt
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2 * Q + 1)
        
        # Vectorized computation for all q
        # q_idx ranges from 0 to 2Q
        q_indices = np.arange(2 * Q + 1)
        q_values = q_indices - Q
        
        # Deltas in value
        # V_next[q+1] - V_next[q]
        # For q=Q, q+1 is out of bounds, but we don't trade bid there.
        # For q=-Q, q-1 is out of bounds, but we don't trade ask there.
        
        # Prepare arrays for vectorization
        # We evaluate the objective for all grid points and all q's
        # Shape: (num_grid, 2Q+1)
        
        # p_grid shape (num_grid, 1)
        p_g = p_grid[:, np.newaxis]
        
        # grid shape (num_grid, 1)
        g_g = grid[:, np.newaxis]
        
        # V_next shape (1, 2Q+1)
        V_n = V_next[np.newaxis, :]
        
        # Delta V+ shape (1, 2Q+1)
        # V_next[q+1] - V_next[q]
        # Note: q index in V_next is q+Q.
        # V_next[q+1] is V_next[q+1+Q] = V_next[q_idx+1]
        # V_next[q] is V_next[q_idx]
        
        # Handle boundaries for Delta V
        # If q_idx == 2Q (q=Q), V_next[q+1] is invalid.
        # We will mask these later.
        
        # Create padded V_next for easy slicing
        V_pad = np.zeros(2 * Q + 3)
        V_pad[1:-1] = V_next
        # V_pad[0] and V_pad[-1] are 0, which is fine for boundary conditions 
        # if we mask them out or if the probability is 0.
        
        # Delta V+ : V(q+1) - V(q)
        DV_plus = V_pad[2:] - V_pad[1:-1]  # Shape (2Q+1,)
        # Delta V- : V(q-1) - V(q)
        DV_minus = V_pad[:-2] - V_pad[1:-1]  # Shape (2Q+1,)
        
        # Objective for Bid:
        # Obj_b(b, q) = p(b) * (b + DV_plus[q]) + (1 - p(b)) * V_next[q]
        # = p(b) * b + p(b) * DV_plus[q] + V_next[q] - p(b) * V_next[q]
        # = p(b) * (b + DV_plus[q] - V_next[q]) + V_next[q]
        # Let C_b[q] = DV_plus[q] - V_next[q] = V_next[q+1] - 2*V_next[q]
        
        C_b = DV_plus - V_next[np.newaxis, :]  # Shape (1, 2Q+1)
        
        # Term to maximize: p_g * (g_g + C_b)
        # Note: V_next[q] is constant w.r.t b, so we can ignore it for argmax.
        Obj_b = p_g * (g_g + C_b)
        
        # Mask boundaries:
        # If q == Q (q_idx == 2Q), no bid fill allowed. Set Obj to -inf.
        # If q == -Q (q_idx == 0), no ask fill allowed (handled in ask).
        
        # Find argmax for each q
        # We want the depth b that maximizes Obj_b for each q.
        
        # Handle q=Q (index 2Q): force depth to inf (or 0 prob)
        # In the table, we can store np.inf or a large number.
        # The problem says "np.inf means no quote".
        
        # For q < Q, find optimal b.
        # We only care about q_idx < 2Q for bid.
        
        # To avoid complex masking, we can just compute argmax and then override boundaries.
        
        # Argmax over axis 0 (grid)
        # If all values are -inf or similar, argmax returns 0.
        
        # We need to be careful with the mask.
        # Let's create a mask for valid bid positions (q < Q)
        valid_bid_mask = np.arange(2 * Q + 1) < 2 * Q
        
        # For invalid positions, we don't care about the argmax, we will set inf later.
        
        # To handle the case where the optimal b is at the boundary of the grid,
        # we just take the grid point.
        
        # Optimization:
        # We can set Obj_b[:, ~valid_bid_mask] = -np.inf
        # But ~valid_bid_mask is only the last column (q=Q).
        
        Obj_b[:, 2 * Q] = -np.inf
        
        bid_indices = np.argmax(Obj_b, axis=0)
        bid_depths = grid[bid_indices]
        
        # Set boundary to inf
        bid_depths[2 * Q] = np.inf
        
        bid_table[n, :] = bid_depths
        
        # --- Ask Side ---
        # Obj_a(a, q) = p(a) * (a + DV_minus[q]) + (1 - p(a)) * V_next[q]
        # C_a[q] = DV_minus[q] - V_next[q] = V_next[q-1] - 2*V_next[q]
        
        C_a = DV_minus - V_next[np.newaxis, :]  # Shape (1, 2Q+1)
        
        Obj_a = p_g * (g_g + C_a)
        
        # Mask q = -Q (index 0)
        Obj_a[:, 0] = -np.inf
        
        ask_indices = np.argmax(Obj_a, axis=0)
        ask_depths = grid[ask_indices]
        
        # Set boundary to inf
        ask_depths[0] = np.inf
        
        ask_table[n, :] = ask_depths
        
        # Update V_curr
        # V_curr[q] = -penalty_coeff * q^2 + max(Obj_b[q], Obj_a[q] + V_next[q]? No.)
        
        # Let's re-calculate the max value properly.
        # Value from Bid: p(b*) * (b* + DV_plus) + (1-p(b*)) * V_next
        # Value from Ask: p(a*) * (a* + DV_minus) + (1-p(a*)) * V_next
        # Wait, the max is over (b, a) jointly.
        # Since b and a are independent in the probability (approx) and the objective is separable,
        # the max is the sum of the max for b and the max for a?
        # No, the state transition is coupled.
        # However, if we assume independence of fills (which is true in the model: "independently"),
        # then the expected value is:
        # E[Reward] = p_b b + p_a a + p_b p_a (b+a) ...
        # This coupling makes it not strictly separable.
        # But for small dt, p_b p_a is very small.
        # Let's check the magnitude.
        # A=140, k=1.5, dt=0.001.
        # Optimal b ~ 1/k = 0.66.
        # p_b ~ 1 - exp(-140 * exp(-1) * 0.001) ~ 1 - exp(-0.052) ~ 0.05.
        # p_b p_a ~ 0.0025.
        # The error is small.
        # Given the time constraint and the "structural change" directive,
        # assuming separability is a good approximation that allows vectorization.
        
        # Let's calculate the max value for each q using the chosen b* and a*.
        
        # V_bid_max[q] = p(b*) * (b* + DV_plus[q]) + (1 - p(b*)) * V_next[q]
        # V_ask_max[q] = p(a*) * (a* + DV_minus[q]) + (1 - p(a*)) * V_next[q]
        
        # Actually, the optimal action is to choose b and a to maximize the sum.
        # If we choose b* maximizing the bid component and a* maximizing the ask component,
        # is that the global max?
        # J(b,a) = J_b(b) + J_a(a) + Coupling(b,a).
        # If Coupling is small, yes.
        
        # Let's compute the values.
        
        # For q where bid is allowed:
        p_b_star = p_grid[bid_indices]
        # For q where bid is inf, p_b_star should be 0.
        # bid_indices for q=Q is 0 (from argmax of -inf), so p_grid[0] is not 0.
        # We need to handle this.
        
        # Create arrays for p_star and depth_star
        p_b_star_safe = np.zeros(2 * Q + 1)
        b_star_safe = np.zeros(2 * Q + 1)
        mask_b = valid_bid_mask
        p_b_star_safe[mask_b] = p_grid[bid_indices[mask_b]]
        b_star_safe[mask_b] = bid_depths[mask_b]
        
        # Value contribution from bid side
        # Note: The term V_next[q] is added regardless.
        # Let's define the "incremental value" from the bid decision relative to not quoting?
        # No, let's just compute the total expected value from the step.
        
        # E[V_next] = p_b V_next(q+1) + p_a V_next(q-1) + (1-p_b-p_a+p_b p_a) V_next(q)
        # This is getting complicated to vectorize if we don't separate.
        
        # Approximation:
        # V_curr[q] = -penalty_coeff * q^2 + V_next[q] + 
        #             p_b_star * (b_star + V_next[q+1] - V_next[q]) +
        #             p_a_star * (a_star + V_next[q-1] - V_next[q])
        
        # This assumes independence and ignores p_b p_a term.
        
        p_a_star_safe = np.zeros(2 * Q + 1)
        a_star_safe = np.zeros(2 * Q + 1)
        mask_a = np.arange(2 * Q + 1) > 0
        p_a_star_safe[mask_a] = p_grid[ask_indices[mask_a]]
        a_star_safe[mask_a] = ask_depths[mask_a]
        
        # Calculate increments
        # For q=Q, V_next[q+1] is V_pad[2Q+2]? No, V_pad is size 2Q+3.
        # V_next[q] is V_pad[q+1].
        # V_next[q+1] is V_pad[q+2].
        # V_next[q-1] is V_pad[q].
        
        # Increment from Bid:
        # p_b * (b + V_next[q+1] - V_next[q])
        # We need V_next[q+1] - V_next[q] = DV_plus
        inc_b = p_b_star_safe * (b_star_safe + DV_plus)
        
        # Increment from Ask:
        # p_a * (a + V_next[q-1] - V_next[q])
        # We need V_next[q-1] - V_next[q] = DV_minus
        inc_a = p_a_star_safe * (a_star_safe + DV_minus)
        
        V_curr = -penalty_coeff * (q_values ** 2) + V_next + inc_b + inc_a
        
        V_next = V_curr
        
        # Check deadline
        if time.time() > deadline:
            # Return what we have so far (partial table)
            # Fill remaining with zeros or inf?
            # If we break, the table is incomplete.
            # We should probably return a default strategy to ensure validity.
            # But the loop goes from N-1 down to 0.
            # If we break early, we have V for later times.
            # We need to fill the earlier times.
            # A safe fallback is the seed strategy: 1/k.
            
            # Let's fill the rest of the table with 1/k
            default_depth = 1.0 / k
            for n_rem in range(n, -1, -1):
                bid_table[n_rem, :] = default_depth
                ask_table[n_rem, :] = default_depth
                # Also need to handle boundaries?
                bid_table[n_rem, 2 * Q] = np.inf
                ask_table[n_rem, 0] = np.inf
                
            break

    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Quote depths based on precomputed HJB tables.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Key for cache
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _HJB_CACHE:
        # Check deadline
        if time.time() > _DEADLINE:
            # Fallback to seed strategy
            depth = np.full(len(inventory), 1.0 / k)
            # Handle boundaries? Seed didn't, but we should for validity?
            # "Values below -3/k... make result invalid".
            # Inf is allowed.
            # Seed returned 1/k for all.
            # If we return 1/k for q=Q, it's a bid quote at inventory limit.
            # The model says "no bid fill at q=Q".
            # If we quote, the probability of fill is forced to 0 by the model?
            # "no bid fill at q=Q".
            # So quoting is harmless but earns nothing.
            # However, the objective includes "depths earned".
            # If no fill, 0 earned.
            # So it's fine.
            
            return depth, depth.copy()
        
        _HJB_CACHE[key] = _solve_hjb(sigma, A, k, gamma, T, steps, Q, _DEADLINE)
    
    bid_table, ask_table = _HJB_CACHE[key]
    
    # Determine current step n
    dt = T / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
        
    # Map inventory to table indices
    # inventory is in [-Q, Q]
    # Table index is q + Q
    # Ensure inventory is within bounds
    inv_clipped = np.clip(inventory, -Q, Q).astype(int)
    indices = inv_clipped + Q
    
    # Retrieve depths
    # bid_table[n, indices]
    bid_depths = bid_table[n, indices]
    ask_depths = ask_table[n, indices]
    
    # Ensure shapes and types
    # bid_depths and ask_depths are arrays of floats.
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
