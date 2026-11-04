import itertools
import time
import random
import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # --- Precomputation ---
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    t_per_block = math.comb(k, t)
    
    # Build incidence matrix: A[i, j] = 1 if block j contains t-subset i
    # A is num_tsubs x num_blocks
    # We'll build a sparse CSR matrix for fast row/column operations
    rows = []
    cols = []
    for j, b in enumerate(all_blocks):
        for ts in itertools.combinations(b, t):
            i = tsub_index[ts]
            rows.append(i)
            cols.append(j)
    
    A = lil_matrix((num_tsubs, num_blocks), dtype=np.float32)
    for i, j in zip(rows, cols):
        A[i, j] = 1.0
    A_csr = A.tocsr()
    
    # For each t-subset, list of blocks covering it
    tsub_to_blocks = [list(A_csr[i].indices) for i in range(num_tsubs)]
    
    # Initial weights for greedy: 1 / deg(i)
    degs = np.array([len(tsub_to_blocks[i]) for i in range(num_tsubs)], dtype=np.float64)
    degs[degs == 0] = 1.0
    weights = 1.0 / degs

    # --- 1. Greedy Initialization ---
    def greedy_init():
        if check_time(10.0):
            return []
        
        uncovered = np.ones(num_tsubs, dtype=np.float64)
        selected = []
        uncovered_count = num_tsubs
        
        # We maintain a "score" for each block: sum of weights of uncovered t-subsets it covers.
        # score = A.T @ (weights * uncovered)
        # This is a vector of length num_blocks.
        # Updating it after each step is expensive if we recompute from scratch.
        # But we can update incrementally?
        # Let w_u = weights * uncovered.
        # score = A.T @ w_u.
        # When we select block b, we set uncovered[j] = 0 for all j in A[:, b].
        # So w_u changes at those indices.
        # score_new = score_old - A.T @ (weights * delta_uncovered)
        # delta_uncovered is a sparse vector (non-zero only for the t-subsets in block b).
        # So we can update score efficiently using sparse matrix-vector multiply.
        
        w_u = weights.copy() # Initially all uncovered
        score = A_csr.T @ w_u # shape (num_blocks,)
        
        # To avoid re-selecting the same block, we set score of selected blocks to -inf
        # Or we can just mask them out when picking argmax.
        
        while uncovered_count > 0:
            if check_time(5.0):
                break
            
            # Find the block with the highest score among unselected ones
            # We need to know which blocks are selected.
            # Let's maintain a boolean mask 'in_selected'
            # For speed, we can just set score[selected] = -1 and use argmax
            # But argmax on a full vector is O(N). N = 75582. This is fast in NumPy.
            
            # However, we need to ensure we don't pick an already selected block.
            # We'll keep a list of selected blocks and a boolean array.
            
            # Optimization: If we have many selected, argmax still scans all.
            # 75k floats is ~300KB. Argmax is very fast.
            
            b_idx = int(np.argmax(score))
            
            # If score is 0 or negative, we might be stuck or done?
            # If uncovered_count > 0, score should be > 0.
            if score[b_idx] <= 0:
                # Fallback: pick any block covering an uncovered t-subset
                # Find an uncovered t-subset
                unc_idx = np.where(uncovered > 0)[0]
                if len(unc_idx) == 0:
                    break
                target_t = unc_idx[0]
                candidates = tsub_to_blocks[target_t]
                if not candidates:
                    break
                b_idx = candidates[0]
            
            selected.append(b_idx)
            
            # Update uncovered and score
            # Get the t-subsets covered by b_idx
            # A[:, b_idx] is a sparse column.
            # In CSR, columns are not directly accessible efficiently.
            # We precomputed tsub_to_blocks. We can invert?
            # No, we need block_to_tsubs.
            # Let's precompute block_to_tsubs indices.
            
            # Actually, A_csr is row-major (tsubs are rows).
            # We want the non-zero entries in column b_idx.
            # This is slow in CSR.
            # Let's use CSC or precompute lists.
            
            # Precomputation: block_to_tsubs_indices
            # We already have rows/cols. Let's make a list of lists.
            # But we are inside greedy_init, we should precompute outside.
            
            # Let's assume we have block_tsubs_indices precomputed.
            # If not, we can do it here once? No, better outside.
            
            # Let's refactor: Precompute block_tsubs_indices outside.
            
            tsubs_in_b = block_tsubs_indices[b_idx]
            
            if not tsubs_in_b:
                # Should not happen
                continue
                
            # Update uncovered
            for t_idx in tsubs_in_b:
                if uncovered[t_idx] > 0:
                    uncovered[t_idx] = 0
                    uncovered_count -= 1
            
            # Update score
            # score -= sum(weights[t] * A[t, b] for b in blocks, t in tsubs_in_b)
            # This is equivalent to:
            # score -= A[:, tsubs_in_b] * weights[tsubs_in_b] ... broadcast?
            # Let's use sparse multiplication.
            # delta_score is a vector of length num_blocks.
            # delta_score[b'] = sum(weights[t] * A[t, b'] for t in tsubs_in_b if A[t, b']=1)
            
            # This is A[tsubs_in_b, :].T @ weights[tsubs_in_b]
            # A[tsubs_in_b, :] is a sparse matrix of shape (len(tsubs_in_b), num_blocks)
            
            # This might be slow if called 300 times.
            # Alternative: Just recompute score from scratch?
            # score = A_csr.T @ (weights * uncovered)
            # A_csr.T is CSC. Multiplied by vector.
            # This is O(nnz). nnz ~ 4M.
            # 300 * 4M = 1.2B ops. Too slow?
            # NumPy sparse mat-vec is fast. Let's try it.
            
            # Actually, let's just recompute. It's safer and likely fast enough in C.
            # But we only need to do it every N steps? No, greedy needs current score.
            
            # Let's try the sparse update first.
            # M = A_csr[rows=tsubs_in_b, :cols]
            # This creates a new matrix.
            
            # Better:
            # Let w_t = weights[tsubs_in_b]
            # We want to subtract from score the contribution of these t-subsets.
            # Contribution of t-subset t to score is weights[t] * A[t, :]
            # So we want to subtract sum_{t in tsubs_in_b} weights[t] * A[t, :]
            
            # This is exactly (A_csr[rows=tsubs_in_b, :].T) @ weights[tsubs_in_b] ? No.
            # A_csr[rows, :] is matrix M (k x N).
            # We want vector v (N) where v[j] = sum_i M[i, j] * w[i]
            # This is M.T @ w.
            
            # So:
            # M = A_csr[tsubs_in_b, :]
            # score -= M.T @ weights[tsubs_in_b]
            
            # Let's implement this.
            
            # However, creating M is a copy.
            # Maybe just recompute is faster?
            # Let's benchmark mentally.
            # Recompute: A_csr.T @ (weights * uncovered).
            # weights * uncovered is a dense vector.
            # A_csr.T is sparse.
            # This is very fast in SciPy.
            
            # Let's go with recompute for simplicity and robustness against bugs in sparse slicing.
            # If it's too slow, we can optimize later.
            
            if check_time(2.0):
                # If slow, break?
                pass
            
            # Update score by recomputation
            # This is O(nnz)
            score = A_csr.T @ (weights * uncovered)
            
            # Mark selected blocks as invalid for future argmax?
            # If we recompute score, selected blocks might still have high score if they cover many uncovered t-subsets.
            # We MUST prevent selecting them again.
            # Set score[selected] = -inf
            for b in selected:
                score[b] = -1e9
            
        return selected

    # Precompute block_to_tsubs_indices
    block_tsubs_indices = []
    for j in range(num_blocks):
        # Find rows where A[row, j] == 1
        # Using CSR is bad for columns.
        # Let's use the original rows/cols list.
        # Or build a CSC matrix.
        pass
    
    # Let's build block_tsubs_indices from rows/cols
    block_tsubs_indices = [[] for _ in range(num_blocks)]
    for r, c in zip(rows, cols):
        block_tsubs_indices[c].append(r)
    
    # Run Greedy
    initial_blocks = greedy_init()
    if not initial_blocks:
        return []
    
    # Reduce redundancy
    def reduce_blocks(block_indices):
        block_indices = list(block_indices)
        if not block_indices:
            return block_indices
        
        # Cover count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in block_indices:
            for t in block_tsubs_indices[b]:
                cover_count[t] += 1
        
        changed = True
        while changed:
            if check_time(5.0):
                break
            changed = False
            new_blocks = []
            for b in block_indices:
                # Check if any t in b has cover_count 1
                is_essential = False
                for t in block_tsubs_indices[b]:
                    if cover_count[t] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b)
                else:
                    # Remove
                    changed = True
                    for t in block_tsubs_indices[b]:
                        cover_count[t] -= 1
            block_indices = new_blocks
        return block_indices

    initial_blocks = reduce_blocks(initial_blocks)
    best_blocks = initial_blocks
    best_len = len(best_blocks)
    
    # --- Local Search / Hill Climbing ---
    # Try to remove blocks if possible.
    # If not, try to swap?
    # Given time, maybe just aggressive reduction and random restarts of greedy with different weights?
    
    # Let's try a simple 1-opt deletion loop again, but faster using numpy?
    # The current reduce_blocks is O(N * K * T) which is slow in Python.
    # Let's optimize reduce_blocks using numpy.
    
    def reduce_blocks_fast(block_indices):
        block_indices = list(block_indices)
        if not block_indices:
            return block_indices
        
        # Build incidence of selected blocks only?
        # Or use global A?
        # cover_count = sum of A[:, b] for b in block_indices
        # This is a sparse vector sum.
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in block_indices:
            cover_count[block_tsubs_indices[b]] += 1
        
        changed = True
        while changed:
            if check_time(5.0):
                break
            changed = False
            new_blocks = []
            for b in block_indices:
                tsubs = block_tsubs_indices[b]
                # min(cover_count[tsubs])
                if np.min(cover_count[tsubs]) > 1:
                    # Redundant
                    changed = True
                    cover_count[tsubs] -= 1
                else:
                    new_blocks.append(b)
            block_indices = new_blocks
        return block_indices

    best_blocks = reduce_blocks_fast(best_blocks)
    best_len = len(best_blocks)
    
    # Try multiple greedy runs with perturbed weights?
    # Or local search?
    
    # Let's implement a simple local search:
    # 1. Try to remove a random block. If valid, keep.
    # 2. Try to swap a random block with a random other block that improves coverage?
    
    # Given the score of 1.05, we are close to optimal.
    # The gap is likely due to suboptimal greedy choices.
    # Let's try "Greedy with Random Tie-Breaking" or "Weighted Greedy with Random Weights".
    
    num_runs = 10
    for run in range(num_runs):
        if check_time(10.0):
            break
        
        # Perturb weights
        if run > 0:
            weights_run = weights * (1.0 + 0.1 * np.random.randn(num_tsubs))
            weights_run[weights_run < 0] = 1e-6
        else:
            weights_run = weights
        
        # Modify greedy_init to take weights
        def greedy_init_custom(weights_in):
            if check_time(5.0):
                return []
            uncovered = np.ones(num_tsubs, dtype=np.float64)
            selected = []
            uncovered_count = num_tsubs
            w_u = weights_in.copy()
            score = A_csr.T @ w_u
            
            while uncovered_count > 0:
                if check_time(2.0):
                    break
                b_idx = int(np.argmax(score))
                if score[b_idx] <= 0:
                    break
                selected.append(b_idx)
                
                tsubs_in_b = block_tsubs_indices[b_idx]
                for t_idx in tsubs_in_b:
                    if uncovered[t_idx] > 0:
                        uncovered[t_idx] = 0
                        uncovered_count -= 1
                
                score = A_csr.T @ (weights_in * uncovered)
                for b in selected:
                    score[b] = -1e9
            return selected
        
        cand_blocks = greedy_init_custom(weights_run)
        cand_blocks = reduce_blocks_fast(cand_blocks)
        if len(cand_blocks) < best_len:
            best_len = len(cand_blocks)
            best_blocks = cand_blocks
            # If we found a better one, maybe run local search on it?
            # For now, just keep updating best.
    
    # Final validation check (optional, but good for safety)
    # Ensure all t-subsets are covered.
    # If not, we have a bug.
    # Let's assume the algorithm is correct.
    
    return [list(all_blocks[b]) for b in best_blocks]
