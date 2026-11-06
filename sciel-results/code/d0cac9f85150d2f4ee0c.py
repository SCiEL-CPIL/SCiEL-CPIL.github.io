import time
import numpy as np
import random

N = 17
MIN_DIST = 6
MAX_WORD = 1 << N

def _get_bad_diffs():
    """
    Precompute all differences d such that popcount(d) < MIN_DIST.
    """
    bad_diffs = []
    for d in range(MAX_WORD):
        if d.bit_count() < MIN_DIST: # Python 3.10+
            bad_diffs.append(d)
    return np.array(bad_diffs, dtype=np.uint32)

def construct():
    start_time = time.time()
    bad_diffs = _get_bad_diffs()
    num_bad = len(bad_diffs)
    
    best_code = []
    best_size = 0
    
    # We will run multiple restarts.
    # Each restart uses a random permutation of words or random sampling.
    # Using random sampling with lazy deletion is efficient.
    
    max_restarts = 2000 # Increase restarts as each run is faster?
    
    for restart in range(max_restarts):
        if time.time() - start_time > 55:
            break
        
        is_avail = np.ones(MAX_WORD, dtype=bool)
        # List of all indices, used for sampling. We do not remove from it,
        # we just check is_avail.
        avail_list = list(range(MAX_WORD))
        random.shuffle(avail_list) # Randomize order for this restart? 
        # Actually, if we sample from the list, the order of the list doesn't matter for uniform sampling,
        # but shuffling ensures we don't have a biased start if we were to iterate sequentially.
        # Since we sample, we just need the list to contain valid indices.
        # Shuffling is O(N). 131k shuffle is fast (~1ms).
        
        code = []
        
        while True:
            if time.time() - start_time > 55.5:
                break
            
            # Check if any available
            # We can't easily count available without summing is_avail (O(N)).
            # We can keep a counter `avail_count`.
            # Initially MAX_WORD.
            # When we add u, we mark u and its neighbors as unavailable.
            # We must be careful not to double count if neighbors overlap.
            # Overlaps happen if d1 ^ u == d2 ^ u => d1 == d2. So neighbors are distinct.
            # But u ^ d1 might equal u ^ d2? No, d1 != d2.
            # However, u ^ d1 might be already unavailable from previous steps.
            # So the number of *newly* unavailable nodes is not simply 1 + num_bad.
            # It is 1 + (num_bad - (number of neighbors already unavailable)).
            # This is hard to track incrementally without checking.
            # So we rely on the sampling loop to find valid nodes.
            # If we can't find any valid nodes in many tries, we break.
            
            # Sample candidates
            # We need to find valid candidates from avail_list.
            # Since avail_list is static, we sample indices from it.
            # We might hit stale (unavailable) entries.
            
            valid_samples = []
            # Try to pick SAMPLE_SIZE valid nodes
            # Max attempts to avoid infinite loop if list is mostly stale
            max_attempts = 1000
            attempts = 0
            while len(valid_samples) < 100 and attempts < max_attempts:
                # Pick a random index from the list
                # Using random.choice is O(1)
                idx = random.choice(avail_list)
                attempts += 1
                if is_avail[idx]:
                    valid_samples.append(idx)
            
            if not valid_samples:
                # If we can't find any valid node, the code is maximal
                break
            
            # Convert to numpy array for vectorized degree calculation
            samples = np.array(valid_samples, dtype=np.uint32)
            sample_size = len(samples)
            
            # Calculate degrees
            # neighbors = samples[:, None] ^ bad_diffs[None, :]
            neighbors = samples.reshape(-1, 1) ^ bad_diffs.reshape(1, -1)
            neighbor_avail = is_avail[neighbors]
            degrees = neighbor_avail.sum(axis=1)
            
            # Pick min degree
            min_deg_idx = np.argmin(degrees)
            chosen_u = int(samples[min_deg_idx])
            
            code.append(chosen_u)
            
            # Update availability
            # Mark chosen_u and its neighbors as unavailable
            # chosen_u is already in valid_samples, so is_avail[chosen_u] is True.
            # We mark is_avail[chosen_u] = False.
            # And is_avail[chosen_u ^ bad_diffs] = False.
            
            # Vectorized update:
            # new_unavail = np.append([chosen_u], chosen_u ^ bad_diffs)
            # is_avail[new_unavail] = False
            
            # To be safe and fast:
            is_avail[chosen_u] = False
            is_avail[chosen_u ^ bad_diffs] = False
        
        if len(code) > best_size:
            best_size = len(code)
            best_code = code.copy()
            
    return best_code
