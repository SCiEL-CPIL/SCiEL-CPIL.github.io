import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute covered t-subsets for each block as lists of indices
    block_covered = []
    for block in all_blocks:
        subs = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[s] for s in subs]
        block_covered.append(indices)
    
    # Build sparse matrix A where A[i,j] = 1 if block i covers t-subset j
    row_indices = []
    col_indices = []
    data = []
    for i, indices in enumerate(block_covered):
        for j in indices:
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
    
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    A_T = A.T.tocsr()
    
    def greedy_reduce(noise_top_k=1, seed=None, do_polish=True):
        if seed is not None:
            random.seed(seed)
            
        active = np.ones(num_t_subsets, dtype=np.float64)
        selected_block_indices = []
        used_blocks = set()
        
        while time.time() < deadline - 8.0:
            gains = A.dot(active)
            for b in used_blocks:
                gains[b] = 0
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            if noise_top_k > 1 and noise_top_k < num_blocks:
                top_indices = np.argsort(gains)[-noise_top_k:]
                b_idx = random.choice(top_indices)
            else:
                b_idx = np.argmax(gains)
                
            selected_block_indices.append(b_idx)
            used_blocks.add(b_idx)
            
            covered_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            active[covered_indices] = 0
            
            if np.all(active == 0):
                break
                
        # Local Search
        if do_polish and time.time() < deadline - 5.0:
            # Deletion pass
            cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
            for b_idx in selected_block_indices:
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] += 1
                
            changed = True
            while changed and time.time() < deadline - 3.0:
                changed = False
                to_remove = []
                for i, b_idx in enumerate(selected_block_indices):
                    covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                    redundant = True
                    for c in covered:
                        if cover_counts[c] == 1:
                            redundant = False
                            break
                    if redundant:
                        to_remove.append(i)
                        
                if to_remove:
                    for i in to_remove:
                        b_idx = selected_block_indices[i]
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_counts[covered] -= 1
                    remove_set = set(to_remove)
                    selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
                    changed = True
                    
            # 1-for-1 Swap (simplified)
            if time.time() < deadline - 3.0:
                improved = True
                while improved and time.time() < deadline - 2.0:
                    improved = False
                    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
                    for b_idx in selected_block_indices:
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_counts[covered] += 1
                    
                    for i, b_idx in enumerate(selected_block_indices):
                        if time.time() > deadline - 2.0:
                            break
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        unique = [c for c in covered if cover_counts[c] == 1]
                        if not unique:
                            continue
                        
                        c0 = unique[0]
                        candidates = A_T.indices[A_T.indptr[c0]:A_T.indptr[c0+1]]
                        
                        best_cand = -1
                        best_score = -1
                        
                        for cand in candidates:
                            if cand == b_idx or cand in used_blocks:
                                continue
                            cand_covered = set(A.indices[A.indptr[cand]:A.indptr[cand+1]])
                            ok = all(u in cand_covered for u in unique)
                            if not ok:
                                continue
                            score = len(cand_covered)
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                                
                        if best_cand != -1:
                            used_blocks.remove(b_idx)
                            used_blocks.add(best_cand)
                            selected_block_indices[i] = best_cand
                            improved = True
                            break
                            
        return selected_block_indices

    best_blocks = None
    best_size = float('inf')
    
    # Run greedy with different noise levels
    configs = [(1, 42), (3, 142), (5, 242), (1, 342)]
    for noise, seed in configs:
        if time.time() > deadline - 15.0:
            break
        res = greedy_reduce(noise_top_k=noise, seed=seed, do_polish=True)
        if res and len(res) < best_size:
            best_size = len(res)
            best_blocks = res
            
    if best_blocks is None:
        best_blocks = greedy_reduce(noise_top_k=1, seed=1, do_polish=True)
        
    blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return blocks
