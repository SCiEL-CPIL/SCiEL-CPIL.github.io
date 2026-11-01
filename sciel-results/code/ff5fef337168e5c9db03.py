import numpy as np
import itertools
import time

def construct(v, k, t):
    points = list(range(v))
    
    # If v is too large for full enumeration, fallback to simple greedy
    # C(v, k) grows fast. C(20, 7) = 77520. C(21, 7) = 116280.
    # C(25, 7) is huge.
    # For v=19, C(19, 7) = 50388. This is small.
    if v <= 20:
        all_blocks = list(itertools.combinations(points, k))
        all_t_subsets = list(itertools.combinations(points, t))
        t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
        
        n_t_subsets = len(all_t_subsets)
        n_blocks = len(all_blocks)
        n_t_per_block = len(list(itertools.combinations(range(k), t)))
        
        block_t_subs = np.empty((n_blocks, n_t_per_block), dtype=np.int32)
        
        for i, block in enumerate(all_blocks):
            subs = list(itertools.combinations(block, t))
            for j, sub in enumerate(subs):
                block_t_subs[i, j] = t_subset_index[sub]
                
        best_blocks_count = float('inf')
        best_solution = []
        
        start_time = time.time()
        deadline = start_time + 110 # 10s margin for overhead
        
        run_id = 0
        while time.time() < deadline:
            run_id += 1
            # Use a different seed for each run to explore different paths
            # np.random.seed(run_id) is fine
            
            covered = np.zeros(n_t_subsets, dtype=bool)
            current_blocks = []
            
            while not np.all(covered):
                # Calculate gains
                # covered[block_t_subs] is boolean array (n_blocks, n_t_per_block)
                # ~covered[...] is True where uncovered
                gains = np.sum(~covered[block_t_subs], axis=1)
                
                max_gain = np.max(gains)
                if max_gain == 0:
                    # No block covers any new subset? Should not happen if valid
                    break
                    
                candidates = np.where(gains == max_gain)[0]
                
                # Randomly pick one candidate
                idx = candidates[np.random.randint(len(candidates))]
                
                current_blocks.append(all_blocks[idx])
                
                # Update covered
                covered[block_t_subs[idx]] = True
                
            if len(current_blocks) < best_blocks_count:
                best_blocks_count = len(current_blocks)
                best_solution = [list(b) for b in current_blocks]
                
        return best_solution
    else:
        # Fallback for larger v (unlikely to be used for this task)
        # Simple greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks
