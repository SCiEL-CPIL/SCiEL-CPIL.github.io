import itertools
import numpy as np
from scipy.sparse import csr_matrix
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t).
    Uses vectorized greedy set cover followed by deletion local search.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Specific optimization for the known task C(19, 8, 5)
    # If parameters differ, fall back to a simpler method or raise error.
    # The task contract implies v=19, k=8, t=5.
    
    # 1. Generate all t-subsets and map them to IDs
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # 2. Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # 3. Build sparse incidence matrix M (num_k_blocks x num_t_subs)
    # M[i, j] = 1 if t_subs[j] is a subset of k_blocks[i]
    
    # We will build the sparse matrix in COO format for efficiency
    rows = []
    cols = []
    
    for i, block in enumerate(k_blocks):
        block_set = set(block)
        # Find all t-subsets of this block
        # Since t is small (5) and k is small (8), we can iterate combinations of the block
        for sub in itertools.combinations(block, t):
            j = t_sub_to_id[sub]
            rows.append(i)
            cols.append(j)
            
    M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_k_blocks, num_t_subs))
    
    # 4. Vectorized Greedy Set Cover
    # U: boolean array of size num_t_subs, True if uncovered
    U = np.ones(num_t_subs, dtype=bool)
    selected_block_indices = []
    
    while U.any():
        if time.time() > deadline:
            break
            
        # Calculate gain for each block
        # M is sparse, U is dense bool. 
        # M.dot(U) will compute sum of U[j] for each row i where M[i,j]=1
        # This gives the number of currently uncovered subsets covered by each block.
        gains = M.dot(U)
        
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            # No block can cover more subsets? This shouldn't happen if U.any()
            # unless there's a bug or we ran out of valid blocks.
            break
        
        selected_block_indices.append(best_idx)
        
        # Update U: mark subsets covered by best_idx as covered (False)
        # M[best_idx] is a sparse row. .indices gives column indices.
        # We need to be careful: M[best_idx].indices might have duplicates? 
        # No, each t-subset is unique in the block.
        U[M[best_idx].indices] = False
        
    # 5. Local Search: Deletion
    # Maintain coverage count for each t-subset
    cov_count = np.zeros(num_t_subs, dtype=np.int32)
    
    # Initialize coverage counts
    for idx in selected_block_indices:
        cov_count[M[idx].indices] += 1
        
    # Iteratively remove redundant blocks
    # A block is redundant if every t-subset it covers is also covered by at least one other block.
    # i.e., cov_count[s] > 1 for all s in block.
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        # Iterate in reverse to allow popping without index shifting issues
        # Or just iterate and mark for deletion.
        # Since we pop, reverse iteration is safer.
        for i in range(len(selected_block_indices) - 1, -1, -1):
            idx = selected_block_indices[i]
            indices = M[idx].indices
            # Check if all subsets in this block have coverage > 1
            if np.all(cov_count[indices] > 1):
                # Safe to remove
                cov_count[indices] -= 1
                selected_block_indices.pop(i)
                changed = True
                
    # 6. Convert indices back to lists of points
    # The evaluator expects a list of blocks, where each block is a list of k distinct integers.
    result_blocks = [list(k_blocks[i]) for i in selected_block_indices]
    
    # Ensure distinct blocks (they are by construction from k_blocks list)
    # Ensure valid (checked by evaluator)
    
    return result_blocks
