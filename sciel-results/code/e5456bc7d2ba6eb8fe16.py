"""Improved heuristic with flexibility-aware scoring for ant colony bin packing."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    # Pairwise sums
    S = d[:, None] + d[None, :]
    
    # Feasibility mask
    feasible = S <= cap
    
    # Gaps
    G = cap - S
    
    # For infeasible pairs, G might be negative, handle carefully
    G = np.where(feasible, G, 0.0)
    
    # Sort demands for efficient counting of items that fit in a gap
    d_sorted = np.sort(d)
    
    # N_fit[gap] = number of items with size <= gap
    # We need this for each pair (i,j)
    # Using searchsorted on the sorted array
    # G is an n x n array. We can flatten, process, and reshape.
    G_flat = G.flatten()
    # Only compute for feasible pairs to avoid negative gaps issues, 
    # though searchsorted handles negative values (returns 0)
    N_fit_flat = np.searchsorted(d_sorted, G_flat, side='right')
    N_fit = N_fit_flat.reshape(n, n)
    
    # Flexibility: log(count + 1)
    # If count is 0, flexibility is 0.
    Flexibility = np.log(N_fit + 1.0)
    
    # Emptiness factor: how much space is left?
    # We want to reward flexibility more when the bin is NOT full.
    # Emptiness = (cap - S) / cap
    Emptiness = np.where(feasible, (cap - S) / cap, 0.0)
    
    # Base heuristic: Sum of sizes (prefers larger items)
    Base = S
    
    # Bonus term:
    # Scaled by flexibility and emptiness.
    # Lambda controls the strength of the flexibility bonus.
    # If Lambda is too high, it might prefer small items with high flexibility over large items.
    # We want to maintain the preference for large items but break ties in favor of flexibility.
    # So the bonus should be smaller than the difference between large and medium sums.
    # Max S is ~150. Min S is ~40. Diff is 110.
    # Max Flexibility is log(121) ~ 4.8.
    # Max Emptiness is ~1.0.
    # Bonus = Lambda * 4.8 * 1.0.
    # If Lambda = 5, Bonus max is 24.
    # This is significant but less than the sum difference.
    
    Lambda = 5.0
    Bonus = Lambda * Flexibility * Emptiness
    
    H = Base + Bonus
    
    # Apply feasibility mask
    H = np.where(feasible, H, 1e-9)
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
