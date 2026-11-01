import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute which t-subsets each block covers
    # Use a compact representation: for each block, store the list of t-subset indices it covers
    block_subs = []
    for block in all_ks:
        subs = list(itertools.combinations(block, t))
        block_subs.append([idx_map[s] for s in subs])
    
    # For fast gain computation, we need to know for each t-subset, which blocks cover it
    # This is the transpose of the incidence matrix
    # But that's too large. Instead, we'll use the block_subs array.
    
    best_solution = None
    best_score = float('inf')
    
    def run_greedy_with_local_search(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            # Compute gain for each block: number of currently uncovered t-subsets it covers
            gains = np.zeros(n_ks, dtype=np.int32)
            for i in range(n_ks):
                subs = block_subs[i]
                gains[i] = np.sum(~covered[subs])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Among blocks with max gain, pick randomly
            max_blocks = np.where(gains == max_gain)[0]
            best_idx = max_blocks[random.randint(0, len(max_blocks) - 1)]
            
            blocks_indices.append(best_idx)
            
            subs_to_cover = block_subs[best_idx]
            prev_state = covered[subs_to_cover]
            new_covers = np.sum(~prev_state)
            remaining -= new_covers
            covered[subs_to_cover] = True
            
        current_blocks = list(blocks_indices)
        
        # Local search: try to remove blocks that are redundant
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 2.0:
                return current_blocks, len(current_blocks)
                
            improved = False
            new_blocks = []
            
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                for s in block_subs[b_idx]:
                    coverage_count[s] += 1
                    
            for b_idx in current_blocks:
                if np.all(coverage_count[block_subs[b_idx]] > 1):
                    improved = True
                    for s in block_subs[b_idx]:
                        coverage_count[s] -= 1
                else:
                    new_blocks.append(b_idx)
                    
            current_blocks = new_blocks
            
        # Local search: try to swap a block for a better one
        if len(current_blocks) > 0:
            improved = True
            max_swap_rounds = 50
            for _ in range(max_swap_rounds):
                if time.time() - start_time > time_limit - 2.0:
                    break
                if not improved:
                    break
                improved = False
                
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current_blocks:
                    for s in block_subs[b_idx]:
                        coverage_count[s] += 1
                
                # Try swapping each block with another
                for i, b_idx in enumerate(current_blocks):
                    if time.time() - start_time > time_limit - 2.0:
                        break
                    
                    # Temporarily remove b_idx
                    for s in block_subs[b_idx]:
                        coverage_count[s] -= 1
                    
                    # Find uncovered t-subsets
                    uncovered = np.where(coverage_count == 0)[0]
                    if len(uncovered) == 0:
                        # Can remove entirely - but we already did that
                        for s in block_subs[b_idx]:
                            coverage_count[s] += 1
                        continue
                    
                    # Find blocks that cover the most uncovered t-subsets
                    best_swap = None
                    best_gain = -1
                    # Sample a few candidate blocks
                    candidates = random.sample(range(n_ks), min(100, n_ks))
                    for cand in candidates:
                        if cand == b_idx or cand in current_blocks:
                            continue
                        gain = 0
                        for s in block_subs[cand]:
                            if coverage_count[s] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = cand
                    
                    if best_swap is not None and best_gain > 0:
                        # Perform swap
                        for s in block_subs[best_swap]:
                            coverage_count[s] += 1
                        current_blocks[i] = best_swap
                        improved = True
                    else:
                        for s in block_subs[b_idx]:
                            coverage_count[s] += 1
        
        return current_blocks, len(current_blocks)
    
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy_with_local_search(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
    
    if best_solution is None:
        blocks, score = run_greedy_with_local_search(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
