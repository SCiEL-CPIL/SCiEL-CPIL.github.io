import numpy as np
import itertools
import time
import random
from math import comb
import scipy.sparse as sp

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build sparse incidence matrix A (blocks x triples)
    # A[i, j] = 1 if block i covers triple j
    row_indices = []
    col_indices = []
    for i, block in enumerate(all_blocks):
        for triple in itertools.combinations(block, t):
            j = triple_to_idx[triple]
            row_indices.append(i)
            col_indices.append(j)
            
    A = sp.csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(n_blocks, n_triples))
    
    # Precompute triple lists for each block for fast local search
    block_triples = [A.indices[A.indptr[i]:A.indptr[i+1]] for i in range(n_blocks)]
    
    # Initial coverage counts
    triple_counts = np.array(A.sum(axis=0)).flatten().astype(np.int32)
    
    def greedy_cover(random_tiebreak=False, weight_power=1.0):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        # Weights: 1 / count
        # Since counts of uncovered triples don't change, we can use static weights
        # But let's stick to the "dynamic" idea if it helps, or just use static.
        # As argued, static is equivalent to dynamic for uncovered triples.
        weights = 1.0 / (triple_counts + 1).astype(np.float64)
        if weight_power != 1.0:
            weights = weights ** weight_power
            
        # Active blocks
        active_blocks = np.ones(n_blocks, dtype=bool)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Calculate gains for active blocks
            # gain = sum(weights[triples_in_block]) for uncovered triples
            # We can use the sparse matrix multiplication
            # But we need to mask out covered triples and inactive blocks
            # weights_covered = weights * uncovered
            # gains = A @ weights_covered
            # Then set gains of inactive blocks to 0
            
            w_active = weights * uncovered
            gains = A @ w_active
            gains[~active_blocks] = 0
            
            if random_tiebreak:
                max_gain = np.max(gains)
                if max_gain <= 1e-12:
                    break
                candidates = np.where(gains >= max_gain - 1e-12)[0]
                candidates = candidates[active_blocks[candidates]]
                if len(candidates) == 0:
                    break
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] <= 1e-12:
                    break
            
            selected.append(best_idx)
            active_blocks[best_idx] = False
            uncovered[block_triples[best_idx]] = False
            
        return selected

    def remove_redundant(block_indices):
        if len(block_indices) == 0:
            return block_indices
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline:
                    return result
                bi = result[i]
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    continue
                coverage[triples_in_block] -= 1
                result.pop(i)
                changed = True
                break
        
        return result

    def local_search(block_indices, max_iterations=50):
        if len(block_indices) == 0:
            return block_indices
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                block_triples_bi = block_triples[bi]
                
                # 1. Check for redundancy
                if np.all(coverage[block_triples_bi] > 1):
                    coverage[block_triples_bi] -= 1
                    current.pop(i)
                    improved = True
                    break
                    
                # 2. Try to swap
                critical_mask = coverage[block_triples_bi] == 1
                if not np.any(critical_mask):
                    continue
                
                critical_triples = block_triples_bi[critical_mask]
                
                # Find candidate blocks that cover ALL critical_triples
                # Pick the rarest critical triple to minimize candidates
                min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
                candidates = A.indices[A.indptr[min_ct]:A.indptr[min_ct+1]] # This is wrong, A is blocks x triples.
                # We need triple_to_blocks.
                
                # Let's build triple_to_blocks from A.T
                # A.T is triples x blocks
                # We can precompute this.
                
            if not improved:
                break
                
        return current

    # Precompute triple_to_blocks for local search
    AT = A.T.tocsr()
    triple_to_blocks = [AT.indices[AT.indptr[i]:AT.indptr[i+1]] for i in range(n_triples)]
    
    def local_search_v2(block_indices, max_iterations=50):
        if len(block_indices) == 0:
            return block_indices
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                block_triples_bi = block_triples[bi]
                
                # 1. Check for redundancy
                if np.all(coverage[block_triples_bi] > 1):
                    coverage[block_triples_bi] -= 1
                    current.pop(i)
                    improved = True
                    break
                    
                # 2. Try to swap
                critical_mask = coverage[block_triples_bi] == 1
                if not np.any(critical_mask):
                    continue
                
                critical_triples = block_triples_bi[critical_mask]
                
                # Find candidate blocks that cover ALL critical_triples
                min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
                candidates = triple_to_blocks[min_ct]
                
                other_critical = critical_triples[critical_triples != min_ct]
                
                valid_candidates = []
                for cand in candidates:
                    if cand == bi:
                        continue
                    if len(other_critical) == 0:
                        valid_candidates.append(cand)
                    else:
                        # Check if cand covers other_critical
                        # Use set intersection or boolean mask
                        # block_triples[cand] is an array of triple indices
                        # We need to check if all other_critical are in block_triples[cand]
                        # This is slow in Python.
                        # Use a boolean array?
                        # For now, simple check
                        all_covered = True
                        for ct in other_critical:
                            # Check if ct is in block_triples[cand]
                            # Since block_triples[cand] is sorted? No.
                            # Let's use a set for block_triples if needed, or just search.
                            # Given small size (35 triples), linear search is okay.
                            if ct not in block_triples[cand]:
                                all_covered = False
                                break
                        if all_covered:
                            valid_candidates.append(cand)
                
                if valid_candidates:
                    # Pick a random valid candidate or the first one
                    best_cand = valid_candidates[0]
                    coverage[block_triples_bi] -= 1
                    coverage[block_triples[best_cand]] += 1
                    current[i] = best_cand
                    improved = True
                    break
            
            if not improved:
                break
                
        return current

    def local_search_remove_two(block_indices, max_iterations=20):
        if len(block_indices) < 2:
            return block_indices
        
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(current)):
                    bi1 = current[i]
                    bi2 = current[j]
                    both_triples = np.union1d(block_triples[bi1], block_triples[bi2])
                    if np.all(coverage[both_triples] > 2):
                        coverage[both_triples] -= 2
                        current.pop(j)
                        current.pop(i)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        return current

    # Run multiple restarts
    best_solution = None
    best_size = float('inf')
    
    np.random.seed(42)
    random.seed(42)
    
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        wp = random.choice([1.0, 1.5, 2.0])
        selected = greedy_cover(random_tiebreak=True, weight_power=wp)
        selected = remove_redundant(selected)
        selected = local_search_v2(selected, max_iterations=30)
        selected = local_search_remove_two(selected, max_iterations=10)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(all_blocks[i]) for i in selected]
            
    return best_solution
