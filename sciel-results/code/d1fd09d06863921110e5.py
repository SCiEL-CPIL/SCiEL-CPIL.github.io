"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = None

# Calibration:
# We adopt a base depth factor y_base = 1.0.
# The single-step optimal depth maximizes d*(1-exp(-k*d)), which gives y* ~ 1.59.
# However, in a multi-step setting with inventory risk, a slightly smaller base depth
# (y_base = 1.0, i.e., d = 1/k) often performs better because it allows for more
# frequent fills and better adaptation to inventory changes, as seen in higher-scoring inspirations.
_BASE_Y = 1.0
_BASE_EXP = np.exp(_BASE_Y)  # ~2.718

def quote(t, inventory, mid, params):
    """
    Inventory-aware market making strategy using a log-depth formula.
    
    This approach combines the structural idea of the log-depth formula (from the current
    program and inspiration 1) with the calibration of a base depth of 1/k (from inspiration 2,
    which scored 1.1756). The log-depth formula provides a smooth, nonlinear response to
    inventory, which is more effective than a linear skew in capturing the optimal trade-off
    between fill probability and inventory risk.
    
    d_bid = (1/k) * ln(exp(y_base) / (1 - x))
    d_ask = (1/k) * ln(exp(y_base) / (1 + x))
    where x = gamma * sigma^2 * t_rem * q
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are very close to the limit, return a safe fallback
    if time.time() - _start_time > 325.0:
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
    
    # Inventory as float
    inv_float = inventory.astype(np.float64)
    
    # Compute the skew argument
    # x = gamma * sigma^2 * t_rem * q
    x = gamma * sigma * sigma * t_rem * inv_float
    
    # Compute depths using log formula with calibrated base
    # d = (1/k) * ln( _BASE_EXP / (1 +/- x) )
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom_bid = 1.0 - x
        denom_ask = 1.0 + x
        
        # If denom <= 0, the log argument is negative or zero, resulting in -inf or nan.
        # In the optimal policy, if the skew is too large, we should not quote on that side.
        # We will compute raw values and then fix invalid ones to inf.
        raw_bid = np.log(_BASE_EXP / denom_bid) / k
        raw_ask = np.log(_BASE_EXP / denom_ask) / k
    
    # Fix invalid values: if denom <= 0, depth should be inf (no quote)
    bid_depths = np.where(denom_bid <= 0, np.inf, raw_bid)
    ask_depths = np.where(denom_ask <= 0, np.inf, raw_ask)
    
    # Clamp to valid range: depth > -3/k
    min_depth = -3.0 / k + 1e-9
    max_depth = 100.0 / k
    
    # Only clamp finite values; leave inf as inf
    finite_bid = np.isfinite(bid_depths)
    finite_ask = np.isfinite(ask_depths)
    
    if np.any(finite_bid):
        bid_depths[finite_bid] = np.clip(bid_depths[finite_bid], min_depth, max_depth)
    if np.any(finite_ask):
        ask_depths[finite_ask] = np.clip(ask_depths[finite_ask], min_depth, max_depth)
    
    # Handle boundary inventory: no bid at Q, no ask at -Q
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    bid_depths[at_max_inventory] = np.inf
    ask_depths[at_min_inventory] = np.inf
    
    # Ensure no NaN or -inf remain (safety net)
    bad_bid = np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths <= -3.0/k)
    bad_ask = np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths <= -3.0/k)
    
    if np.any(bad_bid):
        bid_depths[bad_bid] = min_depth
    if np.any(bad_ask):
        ask_depths[bad_ask] = min_depth
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
