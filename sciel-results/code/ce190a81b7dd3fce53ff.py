"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_OPT_TABLES = {}


def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation with a reduced depth grid.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps

    # Reduced depth grid: use a log-spaced grid from very small to large
    # The optimal depth scales roughly as 1/k, so we use a range around that
    min_depth = 0.01
    max_depth = max(10.0 / k, 5.0)
    num_depths = 64
    depths_grid = np.logspace(np.log10(min_depth), np.log10(max_depth), num_depths)

    # Number of inventory states
    n_inv = 2 * Q + 1  # from -Q to Q

    # V[t, q] = optimal value-to-go from step t with inventory q
    # We'll use backward induction: V[n, q] = 0 for all q
    V = np.zeros((steps + 1, n_inv))

    # Optimal depth choices
    opt_bid = np.full((steps, n_inv), np.inf)
    opt_ask = np.full((steps, n_inv), np.inf)

    for t in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 5.0:
            # Return a fallback: use 1/k for all
            for q_idx in range(n_inv):
                opt_bid[t, q_idx] = 1.0 / k
                opt_ask[t, q_idx] = 1.0 / k
            # Fill remaining steps with 1/k
            for tt in range(t - 1, -1, -1):
                for q_idx in range(n_inv):
                    opt_bid[tt, q_idx] = 1.0 / k
                    opt_ask[tt, q_idx] = 1.0 / k
            _OPT_TABLES[key] = (opt_bid, opt_ask, depths_grid)
            return _OPT_TABLES[key]

        for q_idx in range(n_inv):
            q = q_idx - Q  # actual inventory value

            best_val = -np.inf
            best_bid = 1.0 / k
            best_ask = 1.0 / k

            # Cost for this step
            cost = (gamma / 2.0) * sigma**2 * q**2 * dt

            for bid_depth in depths_grid:
                # Bid fill probability
                p_bid = 0.0
                if q < Q:  # can take bid fills
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_depth) * dt)

                for ask_depth in depths_grid:
                    # Ask fill probability
                    p_ask = 0.0
                    if q > -Q:  # can take ask fills
                        p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_depth) * dt)

                    # Expected immediate reward: depths earned from fills
                    # Bid fill: earn bid_depth, inventory decreases by 1 (q -> q-1)
                    # Ask fill: earn ask_depth, inventory increases by 1 (q -> q+1)
                    # No fill: earn 0, inventory stays

                    # Expected value
                    # P(bid fill, ask fill) = p_bid * p_ask
                    # P(bid fill, no ask) = p_bid * (1 - p_ask)
                    # P(no bid, ask fill) = (1 - p_bid) * p_ask
                    # P(no fill) = (1 - p_bid) * (1 - p_ask)

                    # Next inventory states:
                    # bid fill: q -> q - 1 (index q_idx - 1)
                    # ask fill: q -> q + 1 (index q_idx + 1)
                    # both: q -> q (net zero)
                    # none: q -> q

                    V_next_bid = V[t + 1, q_idx - 1] if q > -Q else 0.0  # q-1 valid if q > -Q
                    V_next_ask = V[t + 1, q_idx + 1] if q < Q else 0.0   # q+1 valid if q < Q
                    V_next_both = V[t + 1, q_idx]  # q stays
                    V_next_none = V[t + 1, q_idx]

                    # Immediate reward
                    reward_bid = bid_depth
                    reward_ask = ask_depth

                    # Expected reward
                    E_reward = (p_bid * p_ask * (reward_bid + reward_ask) +
                               p_bid * (1 - p_ask) * reward_bid +
                               (1 - p_bid) * p_ask * reward_ask)

                    # Expected continuation
                    E_cont = (p_bid * p_ask * V_next_both +
                              p_bid * (1 - p_ask) * V_next_bid +
                              (1 - p_bid) * p_ask * V_next_ask +
                              (1 - p_bid) * (1 - p_ask) * V_next_none)

                    total = E_reward - cost + E_cont

                    if total > best_val:
                        best_val = total
                        best_bid = bid_depth
                        best_ask = ask_depth

            V[t, q_idx] = best_val
            opt_bid[t, q_idx] = best_bid
            opt_ask[t, q_idx] = best_ask

    _OPT_TABLES[key] = (opt_bid, opt_ask, depths_grid)
    return _OPT_TABLES[key]


def quote(t, inventory, mid, params):
    """
    Use pre-computed optimal depth table via backward induction.
    """
    n = len(inventory)
    
    # Get or compute optimal tables
    opt_bid, opt_ask, depths_grid = _get_optimal_table(params)
    
    # Determine current step index
    dt = params["T"] / params["steps"]
    step_idx = int(round(t / dt))
    if step_idx >= params["steps"]:
        step_idx = params["steps"] - 1
    if step_idx < 0:
        step_idx = 0

    Q = params["Q"]
    
    # Map inventory to index
    inv_idx = np.clip(inventory + Q, 0, 2 * Q)
    
    # Get depths
    bid_depths = opt_bid[step_idx, inv_idx]
    ask_depths = opt_ask[step_idx, inv_idx]
    
    # Handle boundary conditions: no bid at q=Q, no ask at q=-Q
    at_max_q = (inventory == Q)
    at_min_q = (inventory == -Q)
    
    bid_depths = np.where(at_max_q, np.inf, bid_depths)
    ask_depths = np.where(at_min_q, np.inf, ask_depths)
    
    # Validate: ensure no values below -3/k, -inf, or NaN
    k = params["k"]
    min_valid = -3.0 / k
    
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < min_valid) | np.isneginf(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < min_valid) | np.isneginf(ask_depths), np.inf, ask_depths)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
