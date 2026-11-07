"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after item i.
    Larger values are more attractive.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Precompute pair sums and fit mask
    pair = demands[:, None] + demands[None, :]
    fits = pair <= capacity
    
    # 1. Tightness Signal
    # Gap is the wasted space. Smaller gap is better.
    # We use 1/(gap+1) to keep values bounded and positive.
    # For fitting pairs, gap >= 0.
    gap = capacity - pair
    # Ensure we don't divide by zero or negative numbers for non-fitting (though we mask later)
    safe_gap = np.where(fits, gap + 1.0, 1.0)
    tightness = 1.0 / safe_gap
    # Normalize tightness to [0, 1] range roughly? Or keep raw?
    # Max tightness is 1/1 = 1. Min is 1/(C+1).
    # Let's scale it to be more significant.
    tightness = tightness ** 2.0 # Sharpen the preference for tight fits

    # 2. Scarcity Signal (Structural Centrality)
    # Count how many items k can fit with item j (d_j + d_k <= C)
    # This is the degree of node j in the compatibility graph.
    # Lower degree means item j is "scarce" or "hard to place".
    # We want to prioritize placing scarce items.
    compat_count = (demands[None, :] + demands[:, None] <= capacity).sum(axis=1)
    # compat_count[j] is the number of items that can be in the same bin as j.
    # Note: This counts j itself if 2*d_j <= C.
    # Inverse scarcity: items with few partners get higher weight.
    scarcity = 1.0 / (compat_count + 1.0)
    # Normalize scarcity to [0, 1]
    scarcity = scarcity / (scarcity.max() + 1e-9)

    # 3. Size Signal
    # Larger items are harder to place later. Prioritize them.
    size_norm = demands / capacity
    # We want larger items to have higher attractiveness when they are the 'next' item (j).
    size_boost = size_norm[None, :] # Shape (1, n), broadcast to (n, n)

    # 4. Flexibility Signal
    # After placing i and j, how much room is left?
    # waste = gap.
    # Count items k that fit in the remaining gap.
    # d[k] <= gap[i,j]
    sorted_demands = np.sort(demands)
    # num_future[i, j] = number of items k with d[k] <= gap[i,j]
    # searchsorted returns the index where the value would be inserted.
    # side='right' gives count of elements <= value.
    num_future = np.searchsorted(sorted_demands, gap, side='right')
    # For non-fitting pairs, gap is negative, searchsorted might behave oddly or return 0.
    # We mask later, so it's safe.
    max_future = num_future.max() if num_future.max() > 0 else 1
    flexibility = num_future / max_future

    # Combine Signals
    # We want to maximize tightness, scarcity of j, and size of j.
    # Flexibility is a secondary benefit (leaving room for others).
    
    # Weights based on empirical tuning from similar tasks:
    # Tightness is the most direct indicator of good packing.
    # Scarcity and Size guide the order to avoid "orphan" large items.
    
    w_tight = 1.0
    w_scarcity = 0.5
    w_size = 0.3
    w_flex = 0.2

    # Base attractiveness for column j (item j is the one being placed)
    # This part depends on j and the pair (i,j)
    
    # Term 1: Tightness of the pair
    term_tight = w_tight * tightness
    
    # Term 2: Scarcity of item j
    term_scarcity = w_scarcity * scarcity[None, :]
    
    # Term 3: Size of item j
    term_size = w_size * size_boost
    
    # Sum of static/dynamic drivers
    base_attractiveness = term_tight + term_scarcity + term_size
    
    # Multiplier for flexibility
    # If we leave a lot of room that can be filled by many items, it's good.
    multiplier = 1.0 + w_flex * flexibility
    
    heuristic = base_attractiveness * multiplier
    
    # Apply fit mask
    # Non-fitting pairs get a very small value
    heuristic = np.where(fits, heuristic, 1e-9)
    
    # Ensure positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
