"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    total_k = len(all_k_subsets)
    
    # For each k-subset, precompute the bitset of t-subsets it covers
    # and the list of t-subset indices
    k_subsets_masks = []
    k_subsets_t_indices = []
    for ks in all_k_subsets:
        mask = 0
        t_indices = []
        for ts in itertools.combinations(ks, t):
            idx = t_to_idx[ts]
            t_indices.append(idx)
            mask |= (1 << idx)
        k_subsets_masks.append(mask)
        k_subsets_t_indices.append(t_indices)
    
    # For each t-subset, precompute which k-subsets cover it (as indices)
    t_to_k_subsets = [[] for _ in range(total_t)]
    for ks_idx in range(total_k):
        for t_idx in k_subsets_t_indices[ks_idx]:
            t_to_k_subsets[t_idx].append(ks_idx)
    
    def greedy_random(seed):
        rng = random.Random(seed)
        uncovered_mask = (1 << total_t) - 1
        blocks = []
        
        while uncovered_mask:
            # Find an uncovered t-subset. Pick one with fewest remaining candidate blocks.
            # To make this efficient, we can pick a random uncovered t-subset and check
            # its candidates, or iterate through all uncovered t-subsets to find the one
            # with minimum candidate count. The latter is O(total_t) per block, which might
            # be okay. total_t is ~11k. Number of blocks is ~300. 11k * 300 = 3.3M ops.
            # Let's try picking a random uncovered t-subset first for speed, or
            # pick the one with minimum candidates if we have time.
            
            # Strategy: pick a random uncovered t-subset
            # Find index of a random bit set in uncovered_mask
            # Python int doesn't have a direct "random bit" operation, but we can
            # convert to string or use a trick.
            # Trick: shift right until we find a 1.
            # Better: use bin() and pick a random position. But bin() is slow for large ints.
            # Alternative: keep a list of uncovered t-subsets.
            
            # Let's switch to a list of uncovered t-subset indices.
            pass
        # This approach with mask is hard to pick random element.
        # Let's stick to the mask for the "is it covered" check, but use a list for picking.
        # Actually, for greedy, we need to pick a pivot t-subset.
        # Let's just pick the first uncovered t-subset (min index) for simplicity and determinism
        # within a run, and use different random seeds for tie-breaking in the block construction.
        
        # Revised greedy_random:
        uncovered_mask = (1 << total_t) - 1
        blocks = []
        
        while uncovered_mask:
            # Find the lowest set bit to get a pivot
            # This gives a deterministic pivot based on the current uncovered set.
            # To add randomness, we can pick a random uncovered t-subset.
            # Let's try picking a random uncovered t-subset.
            # We can do this by generating a random number in [0, total_t) and checking
            # if it's uncovered. If not, try again. This is inefficient if few are uncovered.
            # Better: maintain a list of uncovered indices.
            
            # For now, let's use the "lowest set bit" pivot but randomize the block construction.
            # This is less likely to escape local minima than random pivots.
            # Let's implement random pivot selection properly.
            
            # Get a random uncovered t-subset index
            # We can use the fact that we have the mask.
            # Method: pick a random position, check if bit is set.
            # If not, pick another.
            # To make it faster, we can convert the mask to a list of set bits when the
            # number of uncovered t-subsets is small.
            
            # For simplicity and speed, let's just pick the first uncovered one (lowest index).
            # The randomness will come from how we choose the block (tie-breaking).
            
            # Find index of lowest set bit
            # In Python: (x & -x).bit_length() - 1
            pivot_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
            
            # Find all k-subsets that cover pivot_idx and only have points that make sense?
            # No, any k-subset covering pivot_idx is a candidate.
            # We want to maximize the number of NEW t-subsets covered.
            # New t-subsets = (k_subset_mask & uncovered_mask).bit_count()
            
            best_gain = -1
            best_ks_idx = -1
            best_ks = None
            
            candidates = t_to_k_subsets[pivot_idx]
            
            # To add randomness, we can shuffle candidates or add noise to the gain.
            # Let's add a small random noise to the gain.
            noise_scale = 0.1
            
            for ks_idx in candidates:
                ks_mask = k_subsets_masks[ks_idx]
                gain = (ks_mask & uncovered_mask).bit_count()
                # Add noise
                gain_noise = gain + rng.uniform(0, noise_scale)
                if gain_noise > best_gain:
                    best_gain = gain_noise
                    best_ks_idx = ks_idx
                    best_ks = all_k_subsets[ks_idx]
            
            # Add the best block
            blocks.append(list(best_ks))
            # Update uncovered mask
            uncovered_mask &= ~k_subsets_masks[best_ks_idx]
            
            # Check time
            if time.time() > deadline - 5.0:
                # If we are running out of time, we might return a partial result?
                # No, the result must be valid. We must finish the current run.
                # So we should check time *before* starting a new run.
                pass
        
        return blocks
    
    # The above greedy_random is not fully randomized in pivot selection.
    # Let's improve it to pick a random uncovered pivot.
    
    def greedy_random_v2(seed):
        rng = random.Random(seed)
        uncovered_mask = (1 << total_t) - 1
        blocks = []
        
        while uncovered_mask:
            # Pick a random uncovered t-subset
            # We can do this by picking a random index and checking if it's uncovered.
            # To speed this up, we can keep a list of uncovered indices.
            # But maintaining that list is overhead.
            # Let's try the random check method.
            pivot_idx = -1
            # Try up to 100 times to find an uncovered one
            for _ in range(100):
                candidate = rng.randint(0, total_t - 1)
                if (uncovered_mask >> candidate) & 1:
                    pivot_idx = candidate
                    break
            
            if pivot_idx == -1:
                # Fallback: find lowest set bit
                pivot_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
            
            # Find best k-subset covering pivot_idx
            best_gain = -1
            best_ks_idx = -1
            best_ks = None
            
            candidates = t_to_k_subsets[pivot_idx]
            noise_scale = 0.1
            
            for ks_idx in candidates:
                ks_mask = k_subsets_masks[ks_idx]
                gain = (ks_mask & uncovered_mask).bit_count()
                gain_noise = gain + rng.uniform(0, noise_scale)
                if gain_noise > best_gain:
                    best_gain = gain_noise
                    best_ks_idx = ks_idx
                    best_ks = all_k_subsets[ks_idx]
            
            blocks.append(list(best_ks))
            uncovered_mask &= ~k_subsets_masks[best_ks_idx]
            
        return blocks
    
    best_blocks = None
    best_len = float('inf')
    seed = 0
    
    while time.time() < deadline - 10.0:
        blocks = greedy_random_v2(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        seed += 1
    
    if best_blocks is None:
        # Fallback to a simple deterministic greedy if no time for random runs
        # This shouldn't happen if deadline is far enough
        return greedy_random_v2(0)
    
    return best_blocks
# EVOLVE-BLOCK-END
