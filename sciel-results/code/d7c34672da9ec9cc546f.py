import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block to subsets mapping (indices)
    block_to_subsets = [[] for _ in range(num_blocks)]
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            block_to_subsets[b_idx].append(subset_to_idx[s])

    # Precompute subset to blocks mapping
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx in range(num_blocks):
        for s_idx in block_to_subsets[b_idx]:
            subset_to_blocks[s_idx].append(b_idx)

    def fast_greedy_deletion(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(num_subsets, dtype=bool)
        # Initialize gains
        gains = np.array([len(bs) for bs in block_to_subsets], dtype=np.int32)
        selected = []
        selected_set = set()
        
        # Greedy Phase
        for _ in range(num_blocks):
            if not uncovered.any():
                break
            
            # Randomized selection to explore different solutions
            noise = np.random.uniform(0, 1, num_blocks) * 5.0
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected.append(best_idx)
            selected_set.add(best_idx)
            
            # Update state
            b_subs = block_to_subsets[best_idx]
            for s in b_subs:
                if uncovered[s]:
                    uncovered[s] = False
                    for b in subset_to_blocks[s]:
                        gains[b] -= 1

        # Deletion Phase
        coverage_counts = np.zeros(num_subsets, dtype=np.int32)
        for b in selected:
            for s in block_to_subsets[b]:
                coverage_counts[s] += 1
        
        changed = True
        while changed:
            changed = False
            random.shuffle(selected)
            for i in range(len(selected)):
                b = selected[i]
                redundant = True
                for s in block_to_subsets[b]:
                    if coverage_counts[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_to_subsets[b]:
                        coverage_counts[s] -= 1
                    selected.pop(i)
                    changed = True
                    break
            
        return selected

    start_time = time.time()
    best_solution = None
    best_len = float('inf')
    seed = 42
    
    while time.time() - start_time < 110:
        sol = fast_greedy_deletion(seed)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
        seed += 1
        
        if time.time() - start_time > 108:
            break
            
    if best_solution is None:
        return [list(all_blocks[i]) for i in fast_greedy_deletion(42)]
        
    return [list(all_blocks[i]) for i in best_solution]
