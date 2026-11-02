import itertools
import numpy as np
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    # Precompute t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Generate pool
    random.seed(42)
    pool = set()
    # Structured: Cyclic shifts of random bases
    for _ in range(100):
        base = tuple(sorted(random.sample(range(v), k)))
        for shift in range(v):
            block = tuple(sorted((x + shift) % v for x in base))
            pool.add(block)
    # Random
    for _ in range(10000):
        pool.add(tuple(sorted(random.sample(range(v), k))))
        
    pool = list(pool)
    
    # Precompute indices and sets for pool
    pool_indices = []
    pool_sets = []
    for block in pool:
        indices = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        idx_arr = np.array(indices, dtype=np.int32)
        pool_indices.append(idx_arr)
        pool_sets.append(set(indices))
        
    # Greedy Set Cover
    uncovered = np.ones(n_t_subsets, dtype=np.bool_)
    selected_idx = []
    available = np.ones(len(pool), dtype=np.bool_)
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        best_idx = -1
        best_count = 0
        for i in range(len(pool)):
            if not available[i]:
                continue
            count = np.sum(uncovered[pool_indices[i]])
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_idx == -1:
            break
            
        selected_idx.append(best_idx)
        available[best_idx] = False
        uncovered[pool_indices[best_idx]] = False
        
    # Handle incomplete coverage
    if uncovered.any():
        while uncovered.any():
            if time.time() - start_time > time_limit:
                break
            best_block = None
            best_count = 0
            best_indices = None
            for _ in range(50):
                block = tuple(sorted(random.sample(range(v), k)))
                indices = np.array([t_subset_to_idx[sub] for sub in itertools.combinations(block, t)], dtype=np.int32)
                count = np.sum(uncovered[indices])
                if count > best_count:
                    best_count = count
                    best_block = block
                    best_indices = indices
            if best_block:
                if best_block in pool:
                    idx = pool.index(best_block)
                else:
                    pool.append(best_block)
                    pool_indices.append(best_indices)
                    pool_sets.append(set(best_indices))
                    idx = len(pool) - 1
                selected_idx.append(idx)
                available[idx] = False
                uncovered[best_indices] = False

    blocks = [list(pool[i]) for i in selected_idx]
    
    # Local Search
    coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
    current_indices = []
    current_sets = []
    for b in blocks:
        idxs = np.array([t_subset_to_idx[sub] for sub in itertools.combinations(b, t)], dtype=np.int32)
        current_indices.append(idxs)
        current_sets.append(set(idxs))
        coverage_counts[idxs] += 1
        
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Deletion
        for i in range(len(blocks)):
            if np.all(coverage_counts[current_indices[i]] > 1):
                coverage_counts[current_indices[i]] -= 1
                blocks.pop(i)
                current_indices.pop(i)
                current_sets.pop(i)
                improved = True
                break
                
        # Swap
        if not improved:
            critical_mask = (coverage_counts == 1)
            for i in range(len(blocks)):
                if time.time() - start_time > time_limit:
                    break
                
                counts_in_i = coverage_counts[current_indices[i]]
                crit_in_i = current_indices[i][counts_in_i == 1]
                
                if len(crit_in_i) == 0:
                    continue
                
                crit_set = set(crit_in_i)
                
                best_swap = -1
                best_score = -1
                
                sample_size = min(200, len(pool))
                sample_indices = random.sample(range(len(pool)), sample_size)
                
                for cand_idx in sample_indices:
                    if pool[cand_idx] == tuple(blocks[i]):
                        continue
                    
                    if not crit_set.issubset(pool_sets[cand_idx]):
                        continue
                        
                    score = np.sum(critical_mask[pool_indices[cand_idx]])
                    
                    if score > best_score:
                        best_score = score
                        best_swap = cand_idx
                        
                if best_swap != -1:
                    coverage_counts[current_indices[i]] -= 1
                    coverage_counts[pool_indices[best_swap]] += 1
                    blocks[i] = list(pool[best_swap])
                    current_indices[i] = pool_indices[best_swap].copy()
                    current_sets[i] = pool_sets[best_swap]
                    improved = True
                    break

    # Final deletion
    for _ in range(5):
        deleted = False
        for i in range(len(blocks)):
            if np.all(coverage_counts[current_indices[i]] > 1):
                coverage_counts[current_indices[i]] -= 1
                blocks.pop(i)
                current_indices.pop(i)
                current_sets.pop(i)
                deleted = True
                break
        if not deleted:
            break

    return blocks
