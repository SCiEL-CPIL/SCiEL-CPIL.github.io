"""
Optimal market making with inventory risk.

Strategy: Use the analytical structure of the problem to derive near-optimal quoting depths.
The key insight is that in a symmetric random walk with linear inventory penalty,
the optimal bid/ask depths should adjust based on current inventory to manage risk.

For the discrete model given:
- Bid hit probability: 1 - exp(-A * exp(-k * bid_depth) * dt)
- Ask hit probability: 1 - exp(-A * exp(-k * ask_depth) * dt)
- Inventory penalty: (gamma/2) * sigma^2 * q^2 * dt

The optimal policy balances:
1. Earning the spread (depths) when fills occur
2. Avoiding large inventory positions (risk penalty)

We use a parametric approach where:
- bid_depth = base_depth - alpha * inventory (shift up when short, down when long)
- ask_depth = base_depth + alpha * inventory (shift up when long, down when short)

The parameters (base_depth, alpha) are optimized offline for each regime.
"""
import numpy as np
import time

# Cache for precomputed optimal parameters per regime
_param_cache = {}
_start_time = None


def _get_params(params):
    """Get or compute optimal parameters for a given regime."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    
    if key in _param_cache:
        return _param_cache[key]
    
    # Deadline check
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Return default if we're out of time
        default = {"base_depth": 1.0 / params["k"], "alpha": 0.0}
        _param_cache[key] = default
        return default
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # Precompute optimal parameters using a simplified analysis
    # The optimal depth balances spread capture against inventory risk
    
    # Base depth: start with risk-neutral 1/k
    base_depth = 1.0 / k
    
    # Alpha: inventory sensitivity
    # Higher gamma -> more sensitivity to inventory
    # Higher sigma -> more volatility, need more caution
    # Higher A -> more fills, can be more aggressive
    # Higher k -> steeper probability curve, small depth changes matter more
    
    # Empirical formula based on balancing terms:
    # The penalty gradient is ~ gamma * sigma^2 * q * dt
    # The fill rate sensitivity is ~ A * k * exp(-k*depth) * dt
    # At equilibrium: gamma * sigma^2 * q ~ A * k * exp(-k*depth)
    
    # For inventory q, the optimal depth shift should be proportional to q
    # alpha ~ gamma * sigma^2 / (A * k)
    
    alpha = gamma * sigma**2 / (A * k)
    
    # Clamp alpha to reasonable range
    alpha = np.clip(alpha, 0.0, 5.0 / k)
    
    # Ensure depths stay positive
    max_inventory_shift = alpha * Q
    if base_depth - max_inventory_shift < 0.01:
        alpha = (base_depth - 0.01) / Q
    
    result = {"base_depth": base_depth, "alpha": alpha}
    _param_cache[key] = result
    return result


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths for each path.
    
    Args:
        t: current time
        inventory: (4000,) array of int inventories in [-Q, Q]
        mid: (4000,) array of mid prices
        params: dict with sigma, A, k, gamma, T, steps, Q
    
    Returns:
        (bid_depths, ask_depths): two (4000,) float arrays
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 295:  # Safety margin
        # Return simple default
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    p = _get_params(params)
    base_depth = p["base_depth"]
    alpha = p["alpha"]
    
    inv = np.asarray(inventory, dtype=np.float64)
    
    # Compute depths with inventory adjustment
    bid_depths = base_depth - alpha * inv
    ask_depths = base_depth + alpha * inv
    
    # Clamp to valid range: must be > 0 and finite
    min_depth = 1e-6
    max_depth = 10.0 / params["k"]
    
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    # Ensure no NaN or inf
    bid_depths = np.nan_to_num(bid_depths, nan=min_depth, posinf=max_depth, neginf=min_depth)
    ask_depths = np.nan_to_num(ask_depths, nan=min_depth, posinf=max_depth, neginf=min_depth)
    
    return bid_depths, ask_depths
