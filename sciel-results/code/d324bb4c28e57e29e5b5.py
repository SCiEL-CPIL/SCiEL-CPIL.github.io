import itertools
import numpy as np
import scipy.sparse as sp
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Generate all k-subsets (candidate blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build sparse matrix A of size n_k x n_t
    row_indices = []
    col_indices = []
    
    for i, block in enumerate(k_subs):
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                
    A = sp.csr_matrix((len(row_indices), (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)
    
    # Uncovered vector
    u = np.ones(n_t, dtype=np.int8)
    
    selected_blocks = []
    
    # Greedy Set Cover
    while u.sum() > 0:
        if time.time() > deadline:
            break
            
        gains = A @ u
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        selected_blocks.append(best_idx)
        
        row_start = A.indptr[best_idx]
        row_end = A.indptr[best_idx + 1]
        cols = A.indices[row_start:row_end]
        u[cols] = 0
        
    # Convert indices back to blocks
    blocks = [list(k_subs[i]) for i in selected_blocks]
    
    # Redundancy Removal
    if blocks and time.time() < deadline - 15:
        selected_t_subs = []
        for idx in selected_blocks:
            row_start = A.indptr[idx]
            row_end = A.indptr[idx + 1]
            cols = A.indices[row_start:row_end]
            selected_t_subs.append(cols)
        
        cover_count = np.zeros(n_t, dtype=np.int32)
        for cols in selected_t_subs:
            cover_count[cols] += 1
        
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i in range(len(selected_blocks)):
                cols = selected_t_subs[i]
                if len(cols) > 0:
                    if np.any(cover_count[cols] == 1):
                        continue
                to_remove.append(i)
                
            if to_remove:
                changed = True
                for i in sorted(to_remove, reverse=True):
                    cols = selected_t_subs[i]
                    if len(cols) > 0:
                        cover_count[cols] -= 1
                    selected_blocks.pop(i)
                    selected_t_subs.pop(i)
        
        blocks = [list(k_subs[i]) for i in selected_blocks]
    
    # Local Search: try to improve by swapping blocks
    if blocks and time.time() < deadline - 5:
        n_selected = len(selected_blocks)
        
        # Precompute for each selected block, the set of t-subsets it uniquely covers
        unique_tsubs = []
        for i in range(n_selected):
            cols = selected_t_subs[i]
            unique = [c for c in cols if cover_count[c] == 1]
            unique_tsubs.append(set(unique))
        
        # Try to replace each block with a better one
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n_selected):
                if time.time() > deadline - 2:
                    break
                    
                current_block = selected_blocks[i]
                current_unique = unique_tsubs[i]
                current_tsubs = set(selected_t_subs[i])
                
                # Find candidate blocks that cover all unique t-subsets
                # and have better "value" (cover more total t-subsets)
                best_candidate = -1
                best_score = -1
                
                # This is expensive, so limit the search
                # Only check a subset of blocks
                for j in range(n_k):
                    if j == current_block:
                        continue
                    
                    # Check if candidate covers all unique t-subsets
                    row_start = A.indptr[j]
                    row_end = A.indptr[j + 1]
                    cand_tsubs = set(A.indices[row_start:row_end])
                    
                    if not current_unique.issubset(cand_tsubs):
                        continue
                    
                    # Score: number of t-subsets covered
                    score = len(cand_tsubs)
                    if score > best_score:
                        best_score = score
                        best_candidate = j
                
                if best_candidate != -1 and best_candidate != current_block:
                    # Replace
                    old_tsubs = set(selected_t_subs[i])
                    new_tsubs = set(A.indices[A.indptr[best_candidate]:A.indptr[best_candidate + 1]])
                    
                    # Update cover counts
                    for c in old_tsubs - new_tsubs:
                        cover_count[c] -= 1
                    for c in new_tsubs - old_tsubs:
                        cover_count[c] += 1
                    
                    selected_blocks[i] = best_candidate
                    selected_t_subs[i] = A.indices[A.indptr[best_candidate]:A.indptr[best_candidate + 1]]
                    unique_tsubs[i] = set(c for c in selected_t_subs[i] if cover_count[c] == 1)
                    improved = True
        
        blocks = [list(k_subs[i]) for i in selected_blocks]
    
    return blocks
