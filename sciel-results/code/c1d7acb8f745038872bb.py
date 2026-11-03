import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute all t-subsets and map to index
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    subset_to_idx = {sub: i for i, sub in enumerate(all_subsets)}
    
    # Precompute for each k-subset (block) the list of t-subset indices it covers
    # This allows fast coverage updates
    block_to_subsets = {}
    for block in itertools.combinations(range(v), k):
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(subset_to_idx[sub])
        block_to_subsets[block] = subs

    best_blocks = None
    best_size = float('inf')
    
    def greedy_run(seed):
        random.seed(seed)
        covered_count = np.zeros(num_subsets, dtype=np.int8)
        blocks = []
        
        while True:
            uncovered_indices = np.where(covered_count == 0)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered subset to start the block
            idx = random.choice(uncovered_indices)
            start_subset = all_subsets[idx]
            
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # Check subsets formed by x and t-1 elements from current block
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        s_idx = subset_to_idx[sub]
                        if covered_count[s_idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update coverage
            b_tuple = tuple(block)
            for s_idx in block_to_subsets[b_tuple]:
                covered_count[s_idx] += 1
        
        return blocks, covered_count

    # Run multiple restarts
    # Estimate time per run is small, so we can do many.
    # Let's do up to 1000 restarts or until deadline.
    for i in range(1000):
        if time.time() > deadline:
            break
        
        blocks, covered_count = greedy_run(i)
        
        # Post-processing: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i, block in enumerate(blocks):
                b_tuple = tuple(block)
                redundant = True
                for s_idx in block_to_subsets[b_tuple]:
                    if covered_count[s_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for s_idx in block_to_subsets[b_tuple]:
                        covered_count[s_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks.copy()
            
    if best_blocks is None:
        # Fallback
        blocks, _ = greedy_run(0)
        return blocks
        
    return best_blocks
