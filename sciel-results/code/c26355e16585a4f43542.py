# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute nothing; the policy is a closed-form function of (inventory, params)
# The optimal quoting in this discrete model:
#   bid_depth = 1/k + skew
#   ask_depth = 1/k - skew
# where skew = (gamma / 2) * sigma^2 * inventory / k
# This shifts quotes to reduce inventory: positive inventory widens bid, tightens ask.
# Clamp to valid range: depth must be > -3/k (i.e., > -1 for k=1.5, > -4 for k=0.75)
# and > 0 for meaningful quoting. Use np.inf for no-quote if depth would be invalid.

def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    
    inv = inventory.astype(np.float64)
    
    # Base depth: risk-neutral
    base_depth = 1.0 / k
    
    # Inventory skew: shift quotes to manage inventory risk
    # skew > 0 when inventory > 0: widen bid (bid_depth = base + skew), tighten ask (ask_depth = base - skew)
    skew = 0.5 * gamma * sigma * sigma * inv / k
    
    bid_depths = base_depth + skew
    ask_depths = base_depth - skew
    
    # Clamp: depths must be > -3/k to be valid. Also, negative depth doesn't make sense for quoting.
    # The contract says values below -3/k are invalid. Let's ensure depth > -3/k.
    # Also, if depth is too small (close to 0 or negative), it might mean no quote.
    # But the problem says np.inf means no quote. Let's just clamp to a minimum valid value.
    # Minimum valid depth: just above -3/k. Let's use -3/k + epsilon.
    min_valid = -3.0 / k + 1e-9
    
    # Actually, re-reading: "Values below -3/k, -inf, NaN or a wrong shape make the whole result invalid."
    # So we need bid_depths > -3/k and ask_depths > -3/k.
    # Let's clamp to min_valid.
    
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
