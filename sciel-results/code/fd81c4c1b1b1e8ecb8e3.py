import numpy as np
import itertools
import time
import random

def construct(v, k, t):
        t_subsets = list(itertools.combinations(range(v), t))
        n_t = len(t_subsets)
        subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
        
        # Precompute for t=3: pair_to_subsets[i][j] = list of (x, idx)
        # For generic t, we can compute on the fly, but t=3 is the task.
        if t == 3:
            pair_to_subsets = [[None]*v for _ in range(v)]
            for i in range(v):
                for j in range(i+1, v):
                    lst = []
                    for x in range(v):
                        if x == i or x == j: continue
                        s = tuple(sorted((i, j, x)))
                        idx = subset_to_idx[s]
                        lst.append((x, idx))
                    pair_to_subsets[i][j] = lst
        else:
            pair_to_subsets = None

        best_blocks = None
        best_len = float('inf')
        
        start_time = time.time()
        deadline = start_time + 100 # 100s budget
        
        # Number of restarts
        # Each restart: Greedy construction.
        
        for restart in range(100):
            if time.time() > deadline:
                break
                
            random.seed(restart) # Deterministic restarts
            
            uncovered = np.ones(n_t, dtype=bool)
            blocks = []
            
            while np.any(uncovered):
                # Find an uncovered subset
                # To speed up, we can keep a list of uncovered indices and update it?
                # np.any is fast. np.where is fast.
                
                uncovered_indices = np.where(uncovered)[0]
                if len(uncovered_indices) == 0:
                    break
                
                # Pick a random uncovered subset to seed the block
                # This helps escape local minima compared to always picking the first.
                seed_idx = random.choice(uncovered_indices)
                seed_subset = t_subsets[seed_idx]
                
                # Start block with the seed subset
                block = list(seed_subset)
                
                # Mark these subsets as covered?
                # Not yet, we only mark when the block is complete.
                # But we know this block will cover seed_subset.
                
                # Add remaining k - t points
                while len(block) < k:
                    # Calculate gain for each candidate x
                    # Gain = number of uncovered t-subsets that would be covered by adding x
                    
                    best_gain = -1
                    best_x = -1
                    
                    # Candidates: all points not in block
                    candidates = [x for x in range(v) if x not in block]
                    
                    for x in candidates:
                        # Calculate gain
                        # If t=3, we look at pairs in block.
                        if t == 3:
                            gain = 0
                            # Pairs in block
                            for i in range(len(block)):
                                for j in range(i+1, len(block)):
                                    # Get list of (y, idx) for pair (block[i], block[j])
                                    # We need the entry where y == x
                                    # pair_to_subsets[block[i]][block[j]] contains (x, idx) if x is valid
                                    # We can search for x in this list.
                                    # Since list is small (v-2=22 items), linear search is ok.
                                    # Or we can precompute a dict for faster lookup.
                                    
                                    # Let's precompute pair_to_subset_dict[i][j] = {x: idx}
                                    pass # We'll refactor to use dict
                        else:
                            # Generic
                            gain = 0
                            for rest in itertools.combinations(block, t-1):
                                s = tuple(sorted(rest + (x,)))
                                if s in subset_to_idx:
                                    idx = subset_to_idx[s]
                                    if uncovered[idx]:
                                        gain += 1
                                        
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            
                    block.append(best_x)
                
                # Block is complete. Sort it.
                block.sort()
                blocks.append(block)
                
                # Update uncovered
                for s in itertools.combinations(block, t):
                    idx = subset_to_idx[tuple(s)]
                    uncovered[idx] = False
                    
            # End of greedy construction for this restart
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                
            # Optional: Local Improvement
            # If we have time, try to remove redundant blocks or swap.
            # For now, let's see how good the random-start greedy is.

        return best_blocks
