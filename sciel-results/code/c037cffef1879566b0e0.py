import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    m = len(all_t_subsets)
    
    best_blocks = None
    best_size = float('inf')
    
    # Time limit management
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Number of iterations depends on time available
    # Each iteration involves construction + redundancy check
    # Construction is roughly O(B * k * C(k, t-1) * v)
    # For v=19, k=8, t=5, this is small.
    
    for _ in range(200): # Upper bound on restarts, will break on time
        if time.time() - start_time > time_limit:
            break
        
        random.seed() # Re-seed or just use global random state
        
        cov = [0] * m
        blocks = []
        
        # Greedy Construction
        # We maintain a list of indices of uncovered subsets to find min cov quickly?
        # Actually, just scanning m=11628 items is fast.
        # But we can be smarter.
        
        # Simple loop: find subset with min coverage
        # Initially all 0.
        
        while True:
            # Find the t-subset with minimum coverage
            min_cov_val = min(cov)
            if min_cov_val > 0:
                break
            
            # Find an index with min_cov_val
            # Randomly pick among those with min cov to diversify
            candidates = [i for i in range(m) if cov[i] == min_cov_val]
            idx = random.choice(candidates)
            seed_subset = all_t_subsets[idx]
            
            # Build block greedily
            block = list(seed_subset)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                best_gain = -1
                best_x = -1
                # Candidates are points not in block
                candidates_x = [x for x in range(v) if x not in block]
                
                # Pre-compute combinations of block of size t-1 for gain calculation
                # This depends on current block size
                # We can compute gain for each x
                # Gain(x) = count of S' in C(block, t-1) such that cov[index(S' + x)] == 0
                
                # To speed up, we can iterate over S' first?
                # No, iterating x is better if we cache S'.
                
                block_t_minus_1 = list(itertools.combinations(block, t-1))
                
                for x in candidates_x:
                    gain = 0
                    for s_prime in block_t_minus_1:
                        # Form the t-subset
                        # s_prime is a tuple of t-1 elements
                        # We need to check if s_prime + (x,) is a valid t-subset
                        # It is, since x not in block, so x not in s_prime
                        t_sub = tuple(sorted(s_prime + (x,)))
                        i = subset_to_idx[t_sub]
                        if cov[i] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x != -1:
                        # Random tie breaking
                        if random.random() < 0.5:
                            best_x = x
                
                if best_x == -1:
                    # Should not happen if k <= v and not all covered
                    # But if no gain, pick random
                    best_x = random.choice(candidates_x)
                
                block.append(best_x)
            
            # Sort block for consistency
            block.sort()
            
            # Update coverage
            for t_sub in itertools.combinations(block, t):
                i = subset_to_idx[t_sub]
                cov[i] += 1
            
            blocks.append(block)
        
        # Local Search: Redundancy Removal
        # Repeatedly check if any block can be removed
        # A block can be removed if for every t-subset it covers, 
        # there is at least one other block covering it.
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                block = blocks[i]
                # Check if this block is essential
                is_essential = False
                for t_sub in itertools.combinations(block, t):
                    idx = subset_to_idx[t_sub]
                    if cov[idx] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    # Remove block
                    # Decrement cov
                    for t_sub in itertools.combinations(block, t):
                        idx = subset_to_idx[t_sub]
                        cov[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
        
        # Optional: Try to improve by replacing blocks?
        # Given time constraints, simple redundancy removal is the main gain.
        
    return best_blocks
