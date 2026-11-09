import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    # Sort indices by size descending (stable sort for determinism)
    sorted_indices = np.argsort(-demands, kind='mergesort')
    
    # Run First-Fit-Decreasing
    bins = []  # list of lists of original indices
    bin_remaining = []  # remaining capacity for each bin
    
    for idx in sorted_indices:
        size = demands[idx]
        placed = False
        for b in range(len(bins)):
            if size <= bin_remaining[b]:
                bins[b].append(idx)
                bin_remaining[b] -= size
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_remaining.append(capacity - size)
    
    # Build sparse heuristic matrix: high value for adjacent pairs in FFD bins
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    for bin_items in bins:
        # Set heuristic to 1.0 for adjacent pairs (i, j) in the FFD sequence
        # We make it bidirectional to allow traversal in either direction
        for k in range(len(bin_items) - 1):
            i = bin_items[k]
            j = bin_items[k + 1]
            H[i, j] = 1.0
            H[j, i] = 1.0
            
        # Note: Items at the ends of the chain only have 1 neighbor.
        # Items in the middle have 2.
        # This creates a path graph for each bin.
        
    return H
