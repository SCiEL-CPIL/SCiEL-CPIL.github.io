import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    all_triples = list(itertools.combinations(range(v), t))
    
    num_blocks = len(all_blocks)
    num_triples = len(all_triples)
    
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # 1. Build Sparse Matrix M and block_triples_indices
    row_ind = []
    col_ind = []
    block_triples_indices = [[] for _ in range(num_blocks)]
    
    for b_idx, block in enumerate(all_blocks):
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            row_ind.append(b_idx)
            col_ind.append(t_idx)
            block_triples_indices[b_idx].append(t_idx)
            
    # Convert to numpy arrays for faster indexing in reduce_cover
    block_triples_indices_np = [np.array(ids, dtype=np.int32) for ids in block_triples_indices]
    
    M = sparse.csr_matrix((np.ones(len(row_ind), dtype=np.int8), (row_ind, col_ind)), shape=(num_blocks, num_triples))
    
    # Precompute frequency
    freq = np.asarray(M.sum(axis=0)).flatten().astype(np.float64)
    inv_freq = 1.0 / (freq + 1e-9)
    # Alternative weight
    inv_freq_sqrt = 1.0 / (np.sqrt(freq) + 1e-9)

    best_blocks = None
    best_size = float('inf')
    
    def run_greedy(weight_vec, top_k, seed):
        local_rng = random.Random(seed)
        uncovered = np.ones(num_triples, dtype=np.int8)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                return None
                
            # Gains = M @ (uncovered * weight_vec)
            # Note: uncovered is int8, weight is float64.
            # M is int8.
            # Result is float64.
            weighted_uncovered = (uncovered * weight_vec).astype(np.float64)
            gains = M.dot(weighted_uncovered)
            
            if np.max(gains) == 0:
                break
                
            # Select from top K
            if top_k < num_blocks:
                # Use argpartition for top K (faster than argsort for small K)
                # We want the largest, so we partition by -gains
                # Or just use argsort if K is small relative to N?
                # argsort is O(N log N). argpartition is O(N).
                # For N=346k, argsort is ~30ms. argpartition is ~5ms.
                # Let's use argpartition.
                
                # Get indices of top K
                # np.argpartition(gains, -top_k)[-top_k:] gives the top K unsorted
                top_indices = np.argpartition(gains, -top_k)[-top_k:]
                best_idx = local_rng.choice(top_indices)
            else:
                candidates = np.where(gains == np.max(gains))[0]
                best_idx = local_rng.choice(candidates)
                
            selected.append(best_idx)
            
            # Update uncovered
            uncovered[block_triples_indices_np[best_idx]] = 0
            
        if np.any(uncovered):
            return None
            
        return selected

    def reduce_cover(selected_indices):
        if not selected_indices:
            return []
            
        # Coverage counts
        # Sum the rows of M for selected indices
        # S = M[selected_indices, :]
        # coverage = np.asarray(S.sum(axis=0)).flatten()
        
        # This creates a large temporary array.
        # Alternative:
        coverage = np.zeros(num_triples, dtype=np.int32)
        for idx in selected_indices:
            coverage[block_triples_indices_np[idx]] += 1
            
        current_selected = list(selected_indices)
        changed = True
        while changed:
            changed = False
            new_selected = []
            for b_idx in current_selected:
                idxs = block_triples_indices_np[b_idx]
                # Check if any triple covered by b_idx has coverage 1
                if np.any(coverage[idxs] == 1):
                    new_selected.append(b_idx)
                else:
                    # Remove
                    coverage[idxs] -= 1
                    changed = True
            
            current_selected = new_selected
            
        return current_selected

    # Main Loop
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        
        # Strategy Selection
        # 1. Inv Freq, Top 10
        # 2. Inv Freq, Top 50
        # 3. Inv Freq Sqrt, Top 10
        # 4. Uniform (Inv Freq = 1), Top 10
        
        strategy = restart_count % 4
        
        if strategy == 0:
            w = inv_freq
            top_k = 10
        elif strategy == 1:
            w = inv_freq
            top_k = 50
        elif strategy == 2:
            w = inv_freq_sqrt
            top_k = 10
        else:
            w = np.ones(num_triples, dtype=np.float64)
            top_k = 10
            
        selected = run_greedy(w, top_k, seed=restart_count)
        
        if selected is None:
            continue
            
        reduced = reduce_cover(selected)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_blocks = [list(all_blocks[i]) for i in reduced]
            
            if best_size <= 69:
                break

    if best_blocks is None:
        # Fallback
        w = inv_freq
        selected = run_greedy(w, 1, seed=0)
        if selected:
            best_blocks = [list(all_blocks[i]) for i in selected]
        else:
            best_blocks = []
            
    return best_blocks
