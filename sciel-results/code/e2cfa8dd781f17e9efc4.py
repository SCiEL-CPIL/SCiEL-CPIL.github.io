import itertools
import random
import time
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.optimize import linprog

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # safety margin under 150s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-subsets (C(19,7) = 50388)
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_ksubsets)
    
    # Build coverage matrix: A[i,j] = 1 if t-subset i is covered by k-subset j
    # Using sparse matrix for efficiency
    A_lil = lil_matrix((n_tsub, n_ksub), dtype=np.int8)
    for j, ksub in enumerate(all_ksubsets):
        for ts in itertools.combinations(ksub, t):
            i = tsub_index[ts]
            A_lil[i, j] = 1
    A = A_lil.tocsr()
    
    # Step 1: Greedy set cover (scan all k-subsets each step)
    def greedy_set_cover(A, deadline):
        uncovered = np.ones(A.shape[0], dtype=np.int32)
        used_blocks = []
        n_rows, n_cols = A.shape
        
        while np.any(uncovered > 0):
            if time.time() > deadline - 5:
                break
            
            # Find k-subset that covers most uncovered t-subsets
            # gain[j] = sum of uncovered[i] where A[i,j] = 1
            # Efficient: for each column, count how many uncovered rows it touches
            # Using matrix multiply with uncovered vector
            gains = A.T.dot(uncovered)
            best_j = int(np.argmax(gains))
            if gains[best_j] <= 0:
                break
            
            used_blocks.append(best_j)
            # Mark covered t-subsets
            for i in range(n_rows):
                if A[i, best_j] and uncovered[i] > 0:
                    uncovered[i] -= 1
        
        return used_blocks
    
    # Step 2: Local search - try to remove redundant blocks and swap
    def try_remove_blocks(A, blocks, n_tsub, tsub_index):
        """Remove redundant blocks iteratively."""
        blocks = list(blocks)
        if not blocks:
            return blocks
        
        # Coverage count for each t-subset
        covered = np.zeros(n_tsub, dtype=np.int32)
        for j in blocks:
            for i in A.getcol(j).indices:
                covered[i] += 1
        
        changed = True
        while changed:
            changed = False
            # Try removing each block
            for idx in range(len(blocks) - 1, -1, -1):
                j = blocks[idx]
                # Check if all t-subsets in this block have coverage > 1
                removable = True
                col = A.getcol(j)
                for i in col.indices:
                    if covered[i] <= 1:
                        removable = False
                        break
                if removable:
                    for i in col.indices:
                        covered[i] -= 1
                    blocks.pop(idx)
                    changed = True
        
        return blocks
    
    def local_search(A, blocks, n_tsub, k, v, rng, deadline):
        """Local search with removal and swap operations."""
        blocks = list(blocks)
        
        # Initial removal
        blocks = try_remove_blocks(A, blocks, n_tsub, tsub_index)
        best_len = len(blocks)
        
        while time.time() < deadline - 10:
            improved = False
            n = len(blocks)
            
            if n == 0:
                break
            
            # Try random removal + replacement
            i = rng.randint(0, n - 1)
            j_old = blocks[i]
            
            # Coverage with current blocks
            covered = np.zeros(n_tsub, dtype=np.int32)
            for j in blocks:
                col = A.getcol(j)
                for idx in col.indices:
                    covered[idx] += 1
            
            # Temporarily remove block i
            col_old = A.getcol(j_old)
            for idx in col_old.indices:
                covered[idx] -= 1
            
            # Find uncovered t-subsets
            uncovered_mask = (covered == 0)
            n_uncovered = np.sum(uncovered_mask)
            
            if n_uncovered == 0:
                # Can just remove
                blocks.pop(i)
                improved = True
                continue
            
            # Try to find a replacement block that covers all uncovered
            # Sample candidate k-subsets
            best_new_j = None
            best_score = -1
            
            # The new block must contain all uncovered t-subsets
            # For small n_uncovered, we can be more targeted
            max_attempts = min(500, max(50, 2000 - int((time.time() - start_time) * 10)))
            
            for _ in range(max_attempts):
                # Pick a random k-subset
                candidate = tuple(sorted(rng.sample(range(v), k)))
                cand_j = all_ksubsets.index(candidate) if candidate in all_ksubsets else None
                if cand_j is None:
                    # Find index
                    found = False
                    for jj, kk in enumerate(all_ksubsets):
                        if kk == candidate:
                            cand_j = jj
                            found = True
                            break
                    if not found:
                        continue
                
                # Check if it covers all uncovered t-subsets
                col_new = A.getcol(cand_j)
                covers_all = True
                for idx in np.where(uncovered_mask)[0]:
                    if idx not in set(col_new.indices):
                        covers_all = False
                        break
                
                if covers_all:
                    # Score: how many total t-subsets it covers
                    score = len(col_new.indices)
                    if score > best_score:
                        best_score = score
                        best_new_j = cand_j
            
            if best_new_j is not None:
                blocks[i] = best_new_j
                improved = True
            
            if not improved:
                # Try removal pass
                new_blocks = try_remove_blocks(A, blocks, n_tsub, tsub_index)
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    improved = True
            
            if not improved and time.time() > deadline - 15:
                break
        
        return blocks
    
    # Main: multiple greedy starts with local search
    best_blocks = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # First greedy
    blocks = greedy_set_cover(A, deadline)
    blocks = [all_ksubsets[j] for j in blocks]
    blocks = local_search(A, blocks, n_tsub, k, v, rng, deadline)
    if len(blocks) < best_len:
        best_len = len(blocks)
        best_blocks = list(blocks)
    
    # More random restarts
    trial = 0
    while time.time() < deadline - 20:
        trial += 1
        rng2 = random.Random(trial * 7 + 3)
        
        # Perturbed greedy
        blocks_greedy = greedy_set_cover(A, deadline)
        
        # Add some random perturbation
        if len(blocks_greedy) > 0:
            # Remove a few random blocks
            n_remove = min(3, len(blocks_greedy) // 5)
            for _ in range(n_remove):
                idx = rng2.randint(0, len(blocks_greedy) - 1)
                blocks_greedy.pop(idx)
        
        blocks = [all_ksubsets[j] for j in blocks_greedy]
        
        # Local search
        search_deadline = min(deadline - 5, time.time() + 25)
        blocks = local_search(A, blocks, n_tsub, k, v, rng2, search_deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    if best_blocks is None:
        # Fallback: simple greedy
        blocks = greedy_set_cover(A, deadline)
        best_blocks = [all_ksubsets[j] for j in blocks]
    
    return [list(b) for b in best_blocks]
