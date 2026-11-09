# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    Optimal market making based on Avellaneda-Stoikov model approximation.
    Uses a quadratic value function approximation to derive optimal bid/ask depths.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    n = len(inventory)
    inv = np.asarray(inventory, dtype=np.float64)
    
    # Time remaining
    t_rem = T - t
    
    # Handle gamma = 0 case (risk neutral)
    if gamma < 1e-10:
        # If risk neutral, optimal is to quote as tight as possible?
        # Actually, if gamma=0, no inventory penalty.
        # Optimal is to maximize fill rate, so depth -> 0?
        # But spread cost? No spread cost in objective, only depths earned.
        # So b, a -> 0.
        # However, the model has A exp(-k depth).
        # If depth is 0, fill prob is 1-exp(-A dt).
        # Let's just use a small depth.
        min_depth = 0.01
        bid_depth = np.full(n, min_depth)
        ask_depth = np.full(n, min_depth)
    else:
        # Avellaneda-Stoikov optimal spread and reservation price shift
        # Spread term: 2 / (gamma * sigma^2) * log(1 + gamma * sigma^2 / k)
        # Note: The 'k' in AS model is related to the exponential parameter in fill prob.
        # Our fill prob is 1 - exp(-A exp(-k depth) dt).
        # In AS, fill prob is 1 - exp(-k (S - p) dt) for linear, or similar.
        # The term gamma * sigma^2 / k might need adjustment for our specific A and k.
        # Let's stick to the standard form first, it captures the inventory risk correctly.
        
        # The "effective" k in AS is often just k.
        # The "A" parameter in our model scales the intensity.
        # If A is large, we can quote tighter.
        # The AS formula assumes a specific intensity.
        # Let's adjust the spread term to account for A.
        # If A is larger, we can afford tighter quotes (smaller depth) for same fill prob?
        # Actually, higher A means higher fill prob for same depth.
        # So we can make depth smaller.
        # The term log(1 + gamma sigma^2 / k) comes from balancing risk and fill.
        # Let's assume the standard formula holds with our k.
        
        spread_term = (2.0 / (gamma * sigma**2)) * np.log1p(gamma * sigma**2 / k)
        
        # Inventory shift
        # Shift = q * gamma * sigma^2 * t_rem
        shift = inv * gamma * sigma**2 * t_rem
        
        bid_depth = spread_term + shift
        ask_depth = spread_term - shift
        
        # Clip to valid range
        min_depth = 0.01
        max_depth = 20.0
        bid_depth = np.clip(bid_depth, min_depth, max_depth)
        ask_depth = np.clip(ask_depth, min_depth, max_depth)
        
    # Handle inventory limits
    # No bid fill at q = Q (cannot buy more)
    # No ask fill at q = -Q (cannot sell more)
    bid_depth = np.where(inv >= Q, np.inf, bid_depth)
    ask_depth = np.where(inv <= -Q, np.inf, ask_depth)
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
