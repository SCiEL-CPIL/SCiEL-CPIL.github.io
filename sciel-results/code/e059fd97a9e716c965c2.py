"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for pre-computed policies
_policy_cache = {}
_start_time = None


def _solve_hjb(params):
    """
    Solve the HJB equation for the optimal depth policy using a semi-Lagrangian scheme.
    Returns arrays of optimal bid and ask depths indexed by (time_step, inventory).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    # State: inventory from -Q to Q
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    n_q = len(q_values)
    q_index = {q: i for i, q in enumerate(q_values)}

    # Value function V[t, q] for t = 0, ..., steps
    # V[steps, q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_q), dtype=np.float64)

    # Optimal depths: bid_depth[t, q], ask_depth[t, q]
    bid_depth_opt = np.full((steps, n_q), np.inf, dtype=np.float64)
    ask_depth_opt = np.full((steps, n_q), np.inf, dtype=np.float64)

    # Depth grid for search (fine enough for accuracy)
    # Maximum meaningful depth: beyond this, fill probability is negligible
    max_depth = 10.0 / k  # ~6.67 for k=1.5, ~13.3 for k=0.75
    n_depth = 500
    depth_grid = np.linspace(0.01, max_depth, n_depth, dtype=np.float64)

    # Precompute fill probabilities for each depth in the grid
    # P_bid(d) = 1 - exp(-A * exp(-k*d) * dt)
    # P_ask(d) = same formula
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

    # Semi-Lagrangian backward sweep
    for t in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _start_time > 280:
            break

        for i_q, q in enumerate(q_values):
            q_int = int(round(q))
            best_val = -np.inf
            best_bid_d = np.inf
            best_ask_d = np.inf

            for j_b, d_b in enumerate(depth_grid):
                p_b = fill_probs[j_b]
                # Bid fill: inventory decreases by 1
                q_after_b = q_int - 1
                if q_after_b < -Q:
                    continue  # No bid fill at q = -Q... wait, no bid fill at q=Q means can't buy more
                # Actually: "no bid fill at q = Q" means if q == Q, we cannot take a bid (sell)
                # Let me re-read: "no bid fill at q = Q, no ask fill at q = -Q"
                # So if q == Q, bid fill is not allowed (can't increase inventory beyond Q)
                # If q == -Q, ask fill is not allowed (can't decrease inventory below -Q)
                # Wait: bid fill means someone buys from us? Or we buy?
                # "A fill earns its depth and moves inventory one unit"
                # In market making: bid quote -> when hit, we buy, inventory increases
                # ask quote -> when hit, we sell, inventory decreases
                # "no bid fill at q = Q" -> can't buy if already at max inventory
                # "no ask fill at q = -Q" -> can't sell if already at min inventory
                # So bid fill: q -> q+1, not allowed if q == Q
                # Ask fill: q -> q-1, not allowed if q == -Q

                # Let me redo:
                # Bid fill: we buy 1 unit, inventory q -> q+1
                # Not allowed if q == Q
                if q_int == Q:
                    # Cannot take bid fill
                    p_b_eff = 0.0
                else:
                    p_b_eff = p_b
                    q_after_b = q_int + 1

                for j_a, d_a in enumerate(depth_grid):
                    p_a = fill_probs[j_a]
                    if q_int == -Q:
                        p_a_eff = 0.0
                    else:
                        p_a_eff = p_a
                        q_after_a = q_int - 1

                    # Joint fill probability
                    p_both = p_b_eff * p_a_eff
                    p_bid_only = p_b_eff * (1.0 - p_a_eff)
                    p_ask_only = (1.0 - p_b_eff) * p_a_eff
                    p_none = (1.0 - p_b_eff) * (1.0 - p_a_eff)

                    # Expected value
                    val = 0.0
                    if p_both > 0:
                        # Both fill: bid gives +d_b, ask gives +d_a, inventory: q+1-1 = q
                        val += p_both * (d_b + d_a + V[t + 1, i_q])
                    if p_bid_only > 0:
                        val += p_bid_only * (d_b + V[t + 1, q_index[q_int + 1]])
                    if p_ask_only > 0:
                        val += p_ask_only * (d_a + V[t + 1, q_index[q_int - 1]])
                    if p_none > 0:
                        val += p_none * V[t + 1, i_q]

                    # Risk penalty
                    val -= (gamma / 2.0) * sigma * sigma * q_int * q_int * dt

                    if val > best_val:
                        best_val = val
                        best_bid_d = d_b
                        best_ask_d = d_a

            V[t, i_q] = best_val
            bid_depth_opt[t, i_q] = best_bid_d
            ask_depth_opt[t, i_q] = best_ask_d

    return bid_depth_opt, ask_depth_opt, V


def _get_policy(params):
    """Get pre-computed policy, solving if necessary."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        bid_depth_opt, ask_depth_opt, V = _solve_hjb(params)
        _policy_cache[key] = (bid_depth_opt, ask_depth_opt, V)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths based on HJB-solved optimal policy.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_paths = len(inventory)

    # Determine time step index
    t_step = int(round(t / dt))
    if t_step >= steps:
        t_step = steps - 1

    try:
        bid_depth_opt, ask_depth_opt, V = _get_policy(params)

        # Map inventory to policy indices
        inventory_int = np.clip(inventory, -Q, Q)
        indices = inventory_int + Q  # 0 to 2Q

        bid_depths = bid_depth_opt[t_step, indices]
        ask_depths = ask_depth_opt[t_step, indices]

        # Ensure valid depths (not below -3/k)
        min_valid = -3.0 / k
        bid_depths = np.maximum(bid_depths, max(min_valid, 0.01))
        ask_depths = np.maximum(ask_depths, max(min_valid, 0.01))

        # Replace any remaining inf or nan
        bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_valid), bid_depths, 1.0 / k)
        ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_valid), ask_depths, 1.0 / k)

        return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

    except Exception:
        # Fallback: simple inventory-adjusted depth
        depth = np.full(n_paths, 1.0 / k, dtype=np.float64)
        # Adjust for inventory risk
        inventory_float = inventory.astype(np.float64)
        bid_adj = np.clip(1.0 / k * (1.0 - 0.01 * inventory_float / Q), 0.1 / k, 5.0 / k)
        ask_adj = np.clip(1.0 / k * (1.0 + 0.01 * inventory_float / Q), 0.1 / k, 5.0 / k)
        return bid_adj, ask_adj


# EVOLVE-BLOCK-END
