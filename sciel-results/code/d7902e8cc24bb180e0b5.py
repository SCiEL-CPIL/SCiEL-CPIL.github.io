import time
import numpy as np

N = 17
MIN_DIST = 6
MAX_WORD = 1 << N


def _get_bad_diffs():
    """
    Precompute all differences d such that popcount(d) < MIN_DIST.
    These are the 'bad' differences that create conflicts.
    """
    # Efficiently generate all masks with weight < MIN_DIST
    bad_diffs = []
    # Iterate through all possible masks? No, 2^17 is 131072, which is small enough.
    # But we can be smarter.
    # We need d where popcount(d) <= 5.
    # Let's just iterate and check popcount. It's fast enough for 2^17.
    for d in range(MAX_WORD):
        # Using bit_count is Python 3.10+, but bin().count('1') is portable.
        # For speed, we can use a lookup or numpy.
        if bin(d).count('1') < MIN_DIST:
            bad_diffs.append(d)
    return np.array(bad_diffs, dtype=np.uint32)


def construct():
    start_time = time.time()
    
    # Precompute bad differences
    bad_diffs = _get_bad_diffs()
    num_bad = len(bad_diffs)
    
    # We will use a list to track available indices for faster sampling
    # Initially all are available
    # Using a list allows for easy random sampling and removal (though removal from list is O(n), 
    # we can use a set or just a boolean array and sample from it).
    # A boolean array `is_avail` is best for O(1) lookup and vectorized degree calculation.
    
    best_code = []
    best_size = 0
    max_restarts = 1000 # We will run many restarts
    
    # To ensure we finish within time, we'll check time frequently.
    
    for restart in range(max_restarts):
        if time.time() - start_time > 55:
            break
        
        # Initialize availability
        is_avail = np.ones(MAX_WORD, dtype=bool)
        code = []
        
        # We maintain a list of currently available indices for sampling.
        # Updating this list efficiently is key.
        # Instead of updating a list of all available indices (which is O(N) per step),
        # we can sample from the full range and filter, or maintain a set.
        # Given MAX_WORD is 131072, maintaining a list of available indices is feasible if we do it smartly.
        # However, removing items from a list is O(n).
        # Alternative: Use a boolean array for `is_avail` and sample indices from `np.nonzero(is_avail)`.
        # `np.nonzero` is O(N) but implemented in C, so it's fast.
        
        while True:
            if time.time() - start_time > 55.5:
                break
            
            # Find available indices
            avail_idx = np.nonzero(is_avail)[0]
            if len(avail_idx) == 0:
                break
            
            # Sample candidates for MRV
            # We want to pick the one with fewest conflicts among available words.
            # Sample size trade-off: Larger sample = better choice, but slower step.
            # Small sample = faster step, but worse choice.
            # Let's try a moderate sample size, e.g., 50-200.
            sample_size = min(100, len(avail_idx))
            if sample_size == 0:
                break
            
            samples = np.random.choice(avail_idx, sample_size, replace=False)
            
            # Calculate degree for each sample
            # degree(u) = sum(is_avail[u ^ d] for d in bad_diffs)
            # We can vectorize this:
            # For each u in samples, compute neighbors = u ^ bad_diffs
            # Then sum is_avail[neighbors]
            
            # To do this efficiently in NumPy without explicit Python loops over samples:
            # We can reshape samples to (S, 1) and bad_diffs to (1, B)
            # XOR them to get (S, B) matrix of neighbors
            # Then index is_avail with this matrix
            # Then sum along axis 1
            
            # This creates a large temporary array of size S * B.
            # B is roughly sum(C(17, i) for i=0..5) = 1 + 17 + 136 + 680 + 2380 + 6188 = ~9400.
            # S = 100. 100 * 9400 = 940,000 elements. This is small and fast.
            
            s_shape = (sample_size, 1)
            b_shape = (1, num_bad)
            
            # Compute neighbors: (S, 1) XOR (1, B) -> (S, B)
            neighbors = samples.reshape(s_shape) ^ bad_diffs.reshape(b_shape)
            
            # Get availability of these neighbors
            # is_avail is boolean array. Indexing with (S, B) array returns (S, B) boolean array.
            neighbor_avail = is_avail[neighbors]
            
            # Count available neighbors for each sample
            degrees = neighbor_avail.sum(axis=1)
            
            # Pick the sample with the minimum degree
            min_deg_idx = np.argmin(degrees)
            chosen_u = int(samples[min_deg_idx])
            
            # Add to code
            code.append(chosen_u)
            
            # Update availability
            # Mark chosen_u and all its neighbors as unavailable
            # Neighbors of chosen_u are chosen_u ^ bad_diffs
            chosen_neighbors = chosen_u ^ bad_diffs
            is_avail[chosen_neighbors] = False
            is_avail[chosen_u] = False # Ensure self is marked (0 is in bad_diffs, so it's covered, but good to be explicit)
        
        if len(code) > best_size:
            best_size = len(code)
            best_code = code.copy()
            
            # Optional: If we find a very good code, we might stop early or increase restarts?
            # No, just keep going to try to beat it.
            
    return best_code
