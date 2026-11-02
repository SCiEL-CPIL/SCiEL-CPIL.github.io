import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    def time_ok():
        return time.time() < deadline
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    n_restarts = 50  # number of greedy restarts to try
    
    for restart in range(n_restarts):
        if not time_ok():
            break
        
        random.seed(42 + restart)
        
        covered = [False] * n_t
        covered_count = 0
        blocks = []
        
        # Precompute t-subsets for each possible block? Too many. 
        # Instead, compute on the fly.
        
        # For each point, precompute which t-subsets contain it
        # This helps with gain calculation
        point_tsubs = [[] for _ in range(v)]
        for i, sub in enumerate(all_t_subsets):
            for pt in sub:
                point_tsubs[pt].append(i)
        
        # Greedy construction
        while True:
            if not time_ok():
                break
            
            # Find first uncovered t-subset
            uncovered_idx = -1
            for i in range(n_t):
                if not covered[i]:
                    uncovered_idx = i
                    break
            if uncovered_idx == -1:
                break
            
            seed_subset = all_t_subsets[uncovered_idx]
            block_set = set(seed_subset)
            
            # Extend block to size k greedily
            while len(block_set) < k:
                if not time_ok():
                    break
                
                best_x = None
                best_gain = -1
                best_tie = -1
                candidates = [x for x in all_points if x not in block_set]
                random.shuffle(candidates)
                
                for x in candidates:
                    # Count uncovered t-subsets that would be covered by adding x
                    gain = 0
                    for rest in itertools.combinations(tuple(block_set), t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx[sub]
                        if not covered[idx]:
                            gain += 1
                    
                    if gain > best_gain or (gain == best_gain and random.random() < 0.5):
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block_set.add(best_x)
            
            if len(block_set) < k:
                # Fill with random points
                remaining = [x for x in all_points if x not in block_set]
                random.shuffle(remaining)
                for x in remaining[:k - len(block_set)]:
                    block_set.add(x)
            
            block = tuple(sorted(block_set))
            
            # Check if this block is new
            if block not in blocks:
                # Cover all t-subsets in this block
                for sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[sub]
                    if not covered[idx]:
                        covered[idx] = True
                        covered_count += 1
                blocks.append(block)
        
        # Deletion phase: try to remove redundant blocks
        if time_ok():
            # Build coverage count per t-subset
            tsub_block_count = [0] * n_t
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    tsub_block_count[idx] += 1
            
            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    # Check if block i can be removed
                    removable = True
                    for sub in itertools.combinations(blocks[i], t):
                        idx = t_sub_to_idx[sub]
                        if tsub_block_count[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        # Remove block i
                        for sub in itertools.combinations(blocks[i], t):
                            idx = t_sub_to_idx[sub]
                            tsub_block_count[idx] -= 1
                        blocks.pop(i)
                        changed = True
                        break  # restart scan
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
        
        # If we've reached a good solution, we can stop early
        if best_count <= 140:
            break
    
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    return [list(b) for b in best_blocks]
