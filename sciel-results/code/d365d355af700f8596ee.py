import itertools
import random
import time
import numpy as np
from scipy.sparse import coo_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each t-subset to its index
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Build coverage: for each block, which t-subsets it covers
    block_covered_indices = []
    for b in all_blocks:
        idxs = []
        for sub in itertools.combinations(b, t):
            idxs.append(subset_to_idx[sub])
        block_covered_indices.append(np.array(idxs, dtype=np.int32))
    
    # Build sparse coverage matrix A (n_blocks x n_subsets)
    rows = []
    cols = []
    for bi, covered in enumerate(block_covered_indices):
        for ci in covered:
            rows.append(bi)
            cols.append(ci)
    A = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_blocks, n_subsets)).tocsr()
    
    # Fast fitness evaluation using sparse matrix
    def evaluate_coverage(block_indices):
        """Return a boolean array of which t-subsets are covered."""
        if not block_indices:
            return np.zeros(n_subsets, dtype=bool)
        sub_A = A[np.array(block_indices, dtype=np.int32), :]
        return (sub_A.sum(axis=0).A1 > 0)
    
    # Greedy set cover (fast, using incremental gain computation)
    def greedy_cover(max_blocks=200, time_budget=None):
        """Run greedy set cover, return list of block indices."""
        if time_budget is None:
            time_budget = deadline - time.time()
        
        uncovered = np.ones(n_subsets, dtype=bool)
        selected = []
        gain = A.sum(axis=1).A1.copy()  # gain[i] = number of subsets block i covers
        # Actually gain should be number of UNCOVERED subsets block i covers
        # Initially all uncovered, so gain = number of subsets block i covers
        
        while uncovered.any():
            if time.time() > deadline:
                break
            if len(selected) >= max_blocks:
                break
            
            # Find block with max gain
            # Zero out gains for blocks that don't help
            best_idx = int(np.argmax(gain))
            if gain[best_idx] == 0:
                break
            
            selected.append(best_idx)
            # Update uncovered and gains
            new_covered = block_covered_indices[best_idx]
            uncovered[new_covered] = False
            
            # Update gains: for each other block, reduce gain by overlap with newly covered
            # This is expensive, so instead we recompute gain from scratch occasionally
            # For now, just zero out the selected block's gain
            gain[best_idx] = 0
            
            # Periodically recompute gains (every 10 selections)
            if len(selected) % 10 == 0:
                for bi in range(n_blocks):
                    if gain[bi] == 0 and bi not in selected:
                        continue
                    covered_by_bi = block_covered_indices[bi]
                    gain[bi] = sum(uncovered[i] for i in covered_by_bi)
        
        return selected
    
    # Simulated Annealing with local search
    def simulated_annealing(starting_blocks, n_iterations=5000, time_budget=None):
        """Run SA to minimize number of blocks while maintaining coverage."""
        if time_budget is None:
            time_budget = deadline - time.time()
        
        current = list(starting_blocks)
        current_covered = evaluate_coverage(current)
        current_missing = n_subsets - current_covered.sum()
        
        best = current[:]
        best_missing = current_missing
        best_size = len(current)
        
        T = 1.0
        cooling = 0.995
        rng = np.random.RandomState(42)
        
        for it in range(n_iterations):
            if time.time() > deadline:
                break
            if current_missing == 0 and len(current) <= best_size:
                # Try to remove blocks
                if len(current) > 1 and rng.random() < 0.3:
                    # Try removing a random block
                    rm_idx = rng.randint(0, len(current))
                    trial = current[:rm_idx] + current[rm_idx+1:]
                    trial_covered = evaluate_coverage(trial)
                    trial_missing = n_subsets - trial_covered.sum()
                    if trial_missing <= current_missing:
                        current = trial
                        current_missing = trial_missing
                        if len(current) < best_size:
                            best = current[:]
                            best_size = len(current)
                            best_missing = current_missing
                    continue
            
            # Mutation: add or remove a block
            r = rng.random()
            if r < 0.4:
                # Add a random block
                new_block = rng.randint(0, n_blocks - 1)
                if new_block not in current:
                    current.append(new_block)
            elif r < 0.7 and len(current) > 1:
                # Remove a random block
                rm_idx = rng.randint(0, len(current))
                current.pop(rm_idx)
            else:
                # Swap: remove one, add one
                if len(current) > 1:
                    rm_idx = rng.randint(0, len(current))
                    removed = current.pop(rm_idx)
                    new_block = rng.randint(0, n_blocks - 1)
                    if new_block not in current:
                        current.append(new_block)
                    else:
                        current.insert(rm_idx, removed)
            
            # Evaluate
            current_covered = evaluate_coverage(current)
            current_missing = n_subsets - current_covered.sum()
            
            # Objective: minimize (missing, size)
            # Use a scalar objective
            obj_current = current_missing * 1000 + len(current)
            
            if len(best) > 0:
                best_covered = evaluate_coverage(best)
                best_obj = best_missing * 1000 + len(best)
            else:
                best_obj = float('inf')
            
            if obj_current <= best_obj:
                best = current[:]
                best_missing = current_missing
                best_size = len(current)
            
            # SA acceptance
            if obj_current < best_obj or rng.random() < np.exp(-((obj_current - best_obj) / max(T, 0.01))):
                pass  # already accepted
            
            T *= cooling
        
        return best
    
    # Local search: try to remove blocks that are redundant
    def local_search_remove(blocks, max_iter=1000, time_budget=None):
        """Try to remove redundant blocks."""
        if time_budget is None:
            time_budget = deadline - time.time()
        
        blocks = list(blocks)
        rng = np.random.RandomState(123)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if len(blocks) <= 1:
                break
            
            # Try removing a random block
            rm_idx = rng.randint(0, len(blocks))
            trial = blocks[:rm_idx] + blocks[rm_idx+1:]
            trial_covered = evaluate_coverage(trial)
            if trial_covered.all():
                blocks = trial
        
        return blocks
    
    # Phase 1: Get an initial feasible solution via greedy
    initial = greedy_cover(max_blocks=250, time_budget=20)
    
    # Phase 2: Local search to remove redundant blocks
    improved = local_search_remove(initial, max_iter=5000, time_budget=15)
    
    # Phase 3: Simulated annealing to further optimize
    sa_result = simulated_annealing(improved, n_iterations=20000, time_budget=80)
    
    # Phase 4: Final local search
    final = local_search_remove(sa_result, max_iter=5000, time_budget=10)
    
    # Verify completeness
    if not final:
        # Fallback: full greedy
        final = greedy_cover(max_blocks=300)
    
    covered = evaluate_coverage(final)
    if not covered.all():
        # Need to repair
        uncovered_mask = ~covered
        repair_blocks = list(final)
        while uncovered_mask.any():
            if time.time() > deadline:
                break
            # Find best block to add
            best_gain = -1
            best_idx = -1
            for bi in range(n_blocks):
                covered_by_bi = block_covered_indices[bi]
                gain = sum(uncovered_mask[i] for i in covered_by_bi)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = bi
            if best_gain <= 0:
                break
            repair_blocks.append(best_idx)
            uncovered_mask[block_covered_indices[best_idx]] = False
        final = repair_blocks
    
    # Convert to actual blocks
    result = [list(all_blocks[idx]) for idx in final]
    
    return result
