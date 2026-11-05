import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_len = float('inf')
    
    # Helper for comb
    def n_choose_k(n, r):
        if r < 0 or r > n:
            return 0
        return math.comb(n, r)
        
    init_weight = n_choose_k(v - 2, t - 2)
    
    # Run multiple randomized greedy instances
    for restart in range(100000):  # Large number, will break on time
        if time.time() > deadline:
            break
            
        uncovered = set(all_t_subsets)
        # We use a dict for pair weights to save memory and time
        # Only store pairs that have been modified?
        # Or just store all? 
        # Number of pairs is C(19, 2) = 171. Very small.
        # Let's initialize all to init_weight.
        pair_weights = {}
        # Lazy initialization: .get(key, init_weight)
        
        blocks = []
        seen_blocks = set()
        
        while uncovered:
            # Select starting point
            # Pick a random uncovered t-subset
            # For efficiency, if set is large, we can just pick min? 
            # No, random is better for diversity.
            # list(uncovered) is O(N). N <= 3876.
            start_subset = random.choice(list(uncovered))
            
            block = [start_subset[0]]
            
            # Construct block
            retry_count = 0
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for u in block:
                        p1, p2 = (u, x) if u < x else (x, u)
                        gain += pair_weights.get((p1, p2), init_weight)
                    
                    # Random tie-breaking
                    gain += random.random() * 0.01
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                next_point = random.choice(best_candidates)
                block.append(next_point)
            
            block.sort()
            block_tuple = tuple(block)
            
            # Check for duplicate
            if block_tuple in seen_blocks:
                # If duplicate, we didn't cover anything new.
                # We should probably restart the block construction with a different start?
                # Or just try again?
                # If we just continue, we might loop.
                # Let's force a different start for the next iteration by 
                # effectively "skipping" this block generation.
                # We can just `continue` the outer while loop.
                continue
            
            seen_blocks.add(block_tuple)
            blocks.append(block)
            
            # Identify subsets covered by this block that are currently uncovered
            block_t_subsets = list(itertools.combinations(block, t))
            subs_to_cover = []
            for sub in block_t_subsets:
                if sub in uncovered:
                    subs_to_cover.append(sub)
            
            # Remove from uncovered
            for sub in subs_to_cover:
                uncovered.remove(sub)
                
            # Update weights
            for sub in subs_to_cover:
                for u, v in itertools.combinations(sub, 2):
                    p1, p2 = (u, v) if u < v else (v, u)
                    key = (p1, p2)
                    pair_weights[key] = pair_weights.get(key, init_weight) - 1
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        return []
        
    return best_blocks
