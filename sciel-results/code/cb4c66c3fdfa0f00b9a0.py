import itertools
import random
import time
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin for safety
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute for each block, the list of t-subset indices it covers
    block_covered_indices = []
    block_covered_set = []
    
    for b in all_blocks:
        idxs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_covered_indices.append(idxs)
        block_covered_set.append(set(idxs))
        
    # For fast lookup of which blocks cover a specific t-subset
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx, idxs in enumerate(block_covered_indices):
        for s_idx in idxs:
            subset_to_blocks[s_idx].append(b_idx)
    
    def greedy_construct():
        """Greedy set cover to get a good initial solution."""
        covered = np.zeros(n_subsets, dtype=bool)
        blocks = []
        
        while not np.all(covered):
            if time.time() > deadline - 5:
                break
            
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
                
            # Pick the uncovered subset with fewest covering blocks (most constrained)
            counts = np.array([len(subset_to_blocks[i]) for i in uncovered_indices])
            min_count = counts.min()
            candidates = uncovered_indices[counts == min_count]
            target_subset = random.choice(candidates.tolist())
            
            # Get blocks covering this target
            candidate_blocks = subset_to_blocks[target_subset]
            
            best_gain = -1
            best_block = -1
            for b_idx in candidate_blocks:
                gain = np.sum(~covered[block_covered_indices[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block != -1:
                blocks.append(best_block)
                covered[block_covered_indices[best_block]] = True
            else:
                # Fallback, should not happen
                blocks.append(random.randint(0, n_blocks - 1))
                
        return blocks

    def local_search(blocks, max_time=10):
        """Fast local search using coverage-count array for O(1) redundancy checks."""
        current_blocks = list(blocks)
        start_ls = time.time()
        
        # Maintain coverage count array
        count_arr = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            count_arr[block_covered_indices[b_idx]] += 1
        
        improved = True
        while improved and time.time() < start_ls + max_time:
            improved = False
            n_blocks_cur = len(current_blocks)
            
            # 1. Remove redundant blocks (fast: check if any t-subset has count==1)
            for i in range(n_blocks_cur - 1, -1, -1):
                if time.time() > start_ls + max_time:
                    break
                b_idx = current_blocks[i]
                idxs = block_covered_indices[b_idx]
                # Block is redundant if no t-subset it covers has count == 1
                if not np.any(count_arr[idxs] == 1):
                    current_blocks.pop(i)
                    count_arr[idxs] -= 1
                    improved = True
                    break
            
            # 2. Try 1-for-0 removal via 2-block elimination
            if not improved and len(current_blocks) > 1:
                # Sample a few random pairs for 2-for-1
                n_trials = min(50, len(current_blocks))
                for _ in range(n_trials):
                    if time.time() > start_ls + max_time:
                        break
                    i, j = random.sample(range(len(current_blocks)), 2)
                    b_i = current_blocks[i]
                    b_j = current_blocks[j]
                    idxs_i = block_covered_indices[b_i]
                    idxs_j = block_covered_indices[b_j]
                    set_j = block_covered_set[b_j]
                    set_i = block_covered_set[b_i]
                    
                    # Check if removing both leaves any t-subset uncovered
                    # A t-subset is at risk if:
                    # - covered only by b_i (count==1 and not in b_j)
                    # - covered only by b_j (count==1 and not in b_i)
                    # - covered by both b_i and b_j only (count==2 and in both)
                    at_risk_i = [s for s in idxs_i if count_arr[s] == 1 and s not in set_j]
                    at_risk_j = [s for s in idxs_j if count_arr[s] == 1 and s not in set_i]
                    at_risk_both = [s for s in idxs_i if s in set_j and count_arr[s] == 2]
                    
                    if not at_risk_i and not at_risk_j and not at_risk_both:
                        # Safe to remove both
                        if i > j:
                            i, j = j, i
                        current_blocks.pop(j)
                        current_blocks.pop(i)
                        count_arr[idxs_i] -= 1
                        count_arr[idxs_j] -= 1
                        improved = True
                        break
                    else:
                        # Try 2-for-1: find a block covering all at-risk subsets
                        at_risk = at_risk_i + at_risk_j + at_risk_both
                        if at_risk:
                            # Find blocks covering the first at-risk subset
                            cands = set(subset_to_blocks[at_risk[0]])
                            for s in at_risk[1:]:
                                cands &= set(subset_to_blocks[s])
                                if not cands:
                                    break
                            if cands:
                                # Pick a candidate not already in solution
                                cands.difference_update(set(current_blocks))
                                if cands:
                                    best_c = random.choice(list(cands))
                                    if i > j:
                                        i, j = j, i
                                    current_blocks.pop(j)
                                    current_blocks.pop(i)
                                    current_blocks.append(best_c)
                                    count_arr[idxs_i] -= 1
                                    count_arr[idxs_j] -= 1
                                    count_arr[block_covered_indices[best_c]] += 1
                                    improved = True
                                    break

        return current_blocks

    def calculate_coverage(block_indices):
        covered = np.zeros(n_subsets, dtype=bool)
        for idx in block_indices:
            covered[block_covered_indices[idx]] = True
        return covered

    # Main Loop: Multiple Restarts / Iterated Local Search
    best_solution = None
    best_size = float('inf')
    
    # 1. Initial Greedy
    init_blocks = greedy_construct()
    improved = local_search(init_blocks, max_time=15)
    if len(improved) < best_size:
        best_size = len(improved)
        best_solution = improved[:]
        
    if time.time() > deadline - 30:
        return [list(all_blocks[idx]) for idx in best_solution]
        
    # 2. Iterated Local Search
    current_blocks = best_solution[:]
    
    while time.time() < deadline - 10:
        # Perturb: remove 3-6 random blocks
        n_remove = random.randint(3, 6)
        if len(current_blocks) > n_remove:
            to_remove = set(random.sample(range(len(current_blocks)), n_remove))
            current_blocks = [b for i, b in enumerate(current_blocks) if i not in to_remove]
            
        # Repair with greedy (most-constrained-first)
        covered = calculate_coverage(current_blocks)
        while not np.all(covered):
            if time.time() > deadline - 5:
                break
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            # Pick most constrained uncovered subset (fewest covering blocks)
            counts = np.array([len(subset_to_blocks[i]) for i in uncovered_indices])
            min_count = counts.min()
            candidates = uncovered_indices[counts == min_count]
            target_subset = random.choice(candidates.tolist())
            candidate_blocks = subset_to_blocks[target_subset]
            best_gain = -1
            best_block = -1
            for b_idx in candidate_blocks:
                gain = np.sum(~covered[block_covered_indices[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block != -1:
                current_blocks.append(best_block)
                covered[block_covered_indices[best_block]] = True
            else:
                break
                
        # Local Search
        improved = local_search(current_blocks, max_time=5)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = improved[:]
            current_blocks = best_solution[:]
        else:
            current_blocks = improved[:]

    # Final validation: ensure all t-subsets are covered
    final_covered = calculate_coverage(best_solution)
    if not np.all(final_covered):
        while not np.all(final_covered):
            if time.time() > deadline - 2:
                break
            uncovered = np.where(~final_covered)[0]
            if len(uncovered) == 0:
                break
            # Most constrained first
            counts = np.array([len(subset_to_blocks[i]) for i in uncovered])
            min_count = counts.min()
            candidates = uncovered[counts == min_count]
            target = random.choice(candidates.tolist())
            for b_idx in subset_to_blocks[target]:
                if b_idx not in best_solution:
                    best_solution.append(b_idx)
                    final_covered[block_covered_indices[b_idx]] = True
                    break
            else:
                best_solution.append(random.randint(0, n_blocks - 1))
                final_covered = calculate_coverage(best_solution)

    # Remove duplicates
    best_solution = list(dict.fromkeys(best_solution))

    return [list(all_blocks[idx]) for idx in best_solution]
