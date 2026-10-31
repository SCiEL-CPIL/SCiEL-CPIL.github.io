import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    def elapsed():
        return time.time() - start_time

    # 1. Precompute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # 3. Precompute coverage mask for each k-subset as a NumPy boolean matrix
    ksub_masks = np.zeros((n_ksubs, n_tsubs), dtype=bool)
    
    for j, ks in enumerate(all_ksubs):
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            ksub_masks[j, idx] = True
            
    # Convert to int8 for faster matrix multiplication
    ksub_masks_int8 = ksub_masks.astype(np.int8)
    
    # Helper: vectorized greedy
    def greedy_vectorized(rng):
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        remaining = n_tsubs
        blocks_indices = []
        used = np.zeros(n_ksubs, dtype=bool)
        
        while remaining > 0 and elapsed() < deadline - 10.0:
            # Calculate gains for all blocks
            gains = ksub_masks_int8 @ uncovered
            
            # Mask out used blocks
            gains[used] = 0
            
            best_gain = np.max(gains)
            if best_gain == 0:
                break
                
            # Top-K randomization
            # Find the K-th largest gain
            # If K is large, use partition. If small, sort.
            # Let's use K=10
            K = 10
            if len(gains) > K:
                # np.partition is faster than sort for finding top K
                # We want the K-th largest, so partition at -K
                kth_largest = np.partition(gains, -K)[-K]
                candidates = np.where(gains >= kth_largest)[0]
            else:
                candidates = np.where(gains > 0)[0]
                if not len(candidates):
                    break
            
            if not len(candidates):
                break
                
            # Random selection among candidates
            best_j = candidates[rng.randint(len(candidates))]
            
            blocks_indices.append(best_j)
            used[best_j] = True
            
            # Update uncovered
            uncovered[ksub_masks[best_j]] = 0
            remaining -= best_gain
            
        return blocks_indices
    
    # Helper: remove redundant blocks
    def remove_redundants(indices):
        if not indices:
            return indices
            
        # Initialize coverage counts
        cov_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in indices:
            cov_counts[ksub_masks[idx]] += 1
            
        new_indices = []
        for idx in indices:
            mask = ksub_masks[idx]
            # Check if all tsubs covered by this block are covered by others
            # cov_counts includes this block, so we need > 1
            if np.all(cov_counts[mask] > 1):
                # Redundant
                cov_counts[mask] -= 1
            else:
                new_indices.append(idx)
                
        return new_indices

    # Multi-start
    best_solution = None
    best_size = float('inf')
    
    seed = 0
    while elapsed() < deadline - 5.0:
        rng = random.Random(seed)
        greedy_result = greedy_vectorized(rng)
        
        if greedy_result:
            cleaned_result = remove_redundants(greedy_result)
            
            if len(cleaned_result) < best_size:
                best_size = len(cleaned_result)
                best_solution = cleaned_result
                
        seed += 1

    if best_solution is None:
        return []
    
    final_blocks = [list(all_ksubs[idx]) for idx in best_solution]
    return final_blocks
