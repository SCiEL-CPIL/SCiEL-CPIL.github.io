"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix based on 'Simulated Bin-Packing Frequency Prior'.
    
    Strategy:
    1. Run a fast, high-quality bin packing heuristic (First-Fit-Decreasing, FFD)
       to generate a reference set of bins.
    2. Analyze the reference bins to count how often item j appears in the same
       bin as item i, specifically focusing on the 'last item placed' relationship 
       or simple co-occurrence within the same bin.
    3. The heuristic eta[i, j] is set to the frequency of j appearing in a bin 
       where i was also present, normalized. 
    4. To better guide the 'next item' choice, we weight pairs where i is the 
       largest item in the bin (anchor) or simply use co-occurrence counts.
       
    This approach shifts from local geometric heuristics (gap size) to a 
    global, data-driven prior derived from a known-good packing structure.
    """
    start_time = time.time()
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Safety check for time
    if time.time() - start_time > 280:
        return np.full((n, n), 1e-9)
    
    # 1. Generate a reference packing using First-Fit-Decreasing (FFD)
    # FFD is known to produce near-optimal results for many instances.
    # We simulate the process to extract pair statistics.
    
    # Sort indices by size descending for FFD
    sorted_indices = np.argsort(-demands)
    
    # Bins: list of lists of item indices
    bins = []
    bin_sums = []
    item_to_bin = {}
    
    for idx in sorted_indices:
        size = demands[idx]
        if size > capacity:
            # Item too big, skip or handle separately (task says 20-99, cap 150, so all fit)
            continue
            
        placed = False
        for b_idx in range(len(bins)):
            if bin_sums[b_idx] + size <= capacity:
                bins[b_idx].append(idx)
                bin_sums[b_idx] += size
                item_to_bin[idx] = b_idx
                placed = True
                break
        
        if not placed:
            new_bin_idx = len(bins)
            bins.append([idx])
            bin_sums.append(size)
            item_to_bin[idx] = new_bin_idx
    
    # 2. Analyze pairs
    # We want eta[i, j] to be high if j is a good follow-up to i.
    # In the FFD simulation, items are added in sorted order.
    # However, the ACO agent places items in a different order.
    # A robust static heuristic is the 'co-occurrence' frequency:
    # How often do i and j end up in the same bin?
    # Even better: How often is j in the same bin as i, given i is 'large'?
    
    # Let's compute a co-occurrence matrix.
    # co_occ[i, j] = 1 if i and j are in the same bin, else 0.
    # To make it a frequency prior, we can count how many times this specific
    # combination appears across multiple diverse heuristic runs (e.g., FFD, Best-Fit, Random-Fit)
    # to get a more robust signal.
    
    # Let's run 3 different heuristics to get a consensus prior.
    # Heuristic 1: FFD (already done above)
    # Heuristic 2: Best-Fit Decreasing (BFD)
    # Heuristic 3: First-Fit (FF) without sorting (Random order simulation)
    
    co_occurrence_counts = np.zeros((n, n), dtype=float)
    
    # Helper to extract co-occurrence from a list of bins
    def extract_co_occurrence(bins_list):
        counts = np.zeros((n, n), dtype=float)
        for b in bins_list:
            if len(b) > 1:
                # All pairs in this bin co-occur
                for i in b:
                    for j in b:
                        if i != j:
                            counts[i, j] += 1
        return counts
    
    # Add FFD counts
    co_occurrence_counts += extract_co_occurrence(bins)
    
    # Run BFD
    bins_bfd = []
    bin_sums_bfd = []
    for idx in sorted_indices:
        size = demands[idx]
        best_fit_bin = -1
        best_fit_remaining = float('inf')
        
        for b_idx in range(len(bins_bfd)):
            rem = capacity - bin_sums_bfd[b_idx]
            if size <= rem and rem < best_fit_remaining:
                best_fit_bin = b_idx
                best_fit_remaining = rem
        
        if best_fit_bin != -1:
            bins_bfd[best_fit_bin].append(idx)
            bin_sums_bfd[best_fit_bin] += size
        else:
            bins_bfd.append([idx])
            bin_sums_bfd.append(size)
            
    co_occurrence_counts += extract_co_occurrence(bins_bfd)
    
    # Run FF with random order (seeded for reproducibility in heuristic calc, though 
    # the ACO is seeded per instance, the heuristic is static. We just need a good prior.)
    rng = np.random.RandomState(42)
    random_order = rng.permutation(n)
    
    bins_ff_rand = []
    bin_sums_ff_rand = []
    for idx in random_order:
        size = demands[idx]
        placed = False
        for b_idx in range(len(bins_ff_rand)):
            if bin_sums_ff_rand[b_idx] + size <= capacity:
                bins_ff_rand[b_idx].append(idx)
                bin_sums_ff_rand[b_idx] += size
                placed = True
                break
        
        if not placed:
            bins_ff_rand.append([idx])
            bin_sums_ff_rand.append(size)
            
    co_occurrence_counts += extract_co_occurrence(bins_ff_rand)
    
    # Normalize the counts to get a probability-like distribution
    # We want eta[i, j] to be high if j is frequently packed with i.
    # However, the ACO probability is proportional to pheromone[i,j] * eta[i,j].
    # If eta[i,j] is just co-occurrence, it encourages putting items together that 
    # *were* put together in the heuristic. This is a strong inductive bias.
    
    # To avoid zero values for valid pairs that didn't co-occur in these specific runs,
    # we add a small base value for all valid pairs (sum <= capacity).
    
    pair_sums = demands[:, None] + demands[None, :]
    valid = pair_sums <= capacity
    
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Scale co-occurrence counts
    # Max count is 3 (if present in all 3 heuristics)
    # We want valid pairs that co-occur to have high eta.
    # Valid pairs that do NOT co-occur should have low eta, but not 1e-9 if they are valid?
    # Actually, if a pair is valid but never co-occurred in our good heuristics, 
    # it might be a 'bad' pair or just 'unexplored'. 
    # Given the ACO starts with pheromone 1, a low eta will suppress it.
    # This effectively prunes the search space to 'proven good' combinations.
    
    # Let's set eta[i, j] = 1.0 + co_occurrence_counts[i, j] for valid pairs.
    # This ensures all valid pairs are selectable, but co-occurring ones are 2-4x more attractive.
    
    eta[valid] = 1.0 + co_occurrence_counts[valid]
    
    # Diagonal: self-pairing is invalid/meaningless in this context.
    np.fill_diagonal(eta, 1e-9)
    
    # Final check for finite values
    if not np.all(np.isfinite(eta)):
        eta = np.full((n, n), 1e-9)
        np.fill_diagonal(eta, 1e-9)
        
    return eta

# EVOLVE-BLOCK-END
