"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = None

def quote(t, inventory, mid, params):
    """
    Inventory-aware market making strategy with improved skew calibration.
    Uses finite large depths for 'no quote' to avoid inf issues in objective calculation.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: if we are very close to the limit, return a safe fallback
    # 300s limit. If we've been running for > 295s, return simple symmetric quotes.
    if time.time() - _start_time > 295:
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
    
    # Time remaining
    t_rem = T - t
    
    # Base depth: risk-neutral quote depth
    base_depth = 1.0 / k
    
    # Normalize inventory to [-1, 1]
    inv_norm = inventory.astype(np.float64) / Q
    
    # Calculate skew scale
    risk_factor = gamma * sigma * sigma * t_rem
    skew_scale = risk_factor * Q
    
    # Compute bid and ask depth adjustments
    # If inventory is positive (long), we want to sell (Ask) more aggressively (lower depth)
    # and buy (Bid) less aggressively (higher depth).
    # So Bid depth should increase with inventory, Ask depth should decrease.
    bid_adj = base_depth * (1.0 + skew_scale * inv_norm)
    ask_adj = base_depth * (1.0 - skew_scale * inv_norm)
    
    # Define valid range
    # Constraint: depth > -3/k
    min_depth = -3.0 / k + 0.1
    max_depth = 50.0 / k
    
    bid_depths = np.clip(bid_adj, min_depth, max_depth)
    ask_depths = np.clip(ask_adj, min_depth, max_depth)
    
    # Handle boundary inventory
    # Use np.inf to indicate no quote, which is explicitly supported by the evaluator
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    bid_depths[at_max_inventory] = np.inf
    ask_depths[at_min_inventory] = np.inf
    
    # Final safety check: ensure no NaN, -inf, or values <= -3/k
    invalid_mask_bid = np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths <= -3.0/k)
    invalid_mask_ask = np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths <= -3.0/k)
    
    if np.any(invalid_mask_bid):
        bid_depths[invalid_mask_bid] = min_depth
    if np.any(invalid_mask_ask):
        ask_depths[invalid_mask_ask] = min_depth
        
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
