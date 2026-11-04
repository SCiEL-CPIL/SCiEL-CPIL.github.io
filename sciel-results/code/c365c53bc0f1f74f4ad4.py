import itertools
import random
import time
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
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
    block_covered_set = []  # Python set for O(1) membership
    for b in all_blocks:
        idxs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_covered_indices.append(idxs)
        block_covered_set.append(set(idxs))
        
    # For fast lookup of which blocks cover a specific t-subset
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx, idxs in enumerate(block_covered_indices):
        for s_idx in idxs:
            subset_to_blocks[s_idx].append(b_idx)

    def calculate_coverage(block_indices):
        covered = np.zeros(n_subsets, dtype=bool)
        for idx in block_indices:
            covered[block_covered_indices[idx]] = True
        return covered

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
                blocks.append(random.randint(0, n_blocks - 1))
                
        return blocks

    def local_search(blocks, max_time=10):
        """Try to reduce the number of blocks."""
        current_blocks = list(blocks)
        start_ls = time.time()
        
        while time.time() < start_ls + max_time:
            # 1. Remove redundant blocks using count array
            count_arr = np.zeros(n_subsets, dtype=int)
            for b_idx in current_blocks:
                count_arr[block_covered_indices[b_idx]] += 1
            
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_idx = current_blocks[i]
                redundant = True
                for s_idx in block_covered_indices[b_idx]:
                    if count_arr[s_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    current_blocks.pop(i)
                    count_arr[block_covered_indices[b_idx]] -= 1
                    changed = True
                    break
            
            if changed:
                continue
            
            # 2. Try 2-for-1 exchange
            if len(current_blocks) > 2:
                i, j = random.sample(range(len(current_blocks)), 2)
                b_i = current_blocks[i]
                b_j = current_blocks[j]
                
                # Find critical subsets: those that would become uncovered if we remove b_i and b_j
                critical_subs = []
                set_i = block_covered_set[b_i]
                set_j = block_covered_set[b_j]
                
                # Subsets in b_i
                for s_idx in block_covered_indices[b_i]:
                    c = count_arr[s_idx]
                    if s_idx in set_j:
                        new_c = c - 2
                    else:
                        new_c = c - 1
                    if new_c < 1:
                        critical_subs.append(s_idx)
                
                # Subsets in b_j but not in b_i
                for s_idx in block_covered_indices[b_j]:
                    if s_idx in set_i:
                        continue
                    c = count_arr[s_idx]
                    new_c = c - 1
                    if new_c < 1:
                        critical_subs.append(s_idx)
                
                if not critical_subs:
                    # Can remove both without replacement
                    # Remove the higher index first to keep indices valid
                    if i > j:
                        i, j = j, i
                    current_blocks.pop(j)
                    current_blocks.pop(i)
                    break
                else:
                    # Need a block C that covers all critical_subs
                    # First, find the union of points in critical subsets to constrain C
                    # But simpler: just check all blocks
                    crit_set = set(critical_subs)
                    best_c = -1
                    for c_idx in range(n_blocks):
                        if crit_set.issubset(block_covered_set[c_idx]):
                            best_c = c_idx
                            break
                    
                    if best_c != -1:
                        # Replace i, j with best_c
                        if best_c not in current_blocks:
                            # Remove i and j, add best_c
                            if i > j:
                                i, j = j, i
                            current_blocks.pop(j)
                            current_blocks.pop(i)
                            current_blocks.append(best_c)
                        else:
                            # best_c already in set, just remove i and j
                            if i > j:
                                i, j = j, i
                            current_blocks.pop(j)
                            current_blocks.pop(i)
                        break
            
            # If we got here, no change was made
            # Small chance to perturb: remove a random block and let repair handle it
            if time.time() - start_ls > max_time * 0.5 and random.random() < 0.05:
                if len(current_blocks) > 5:
                    rm_idx = random.randint(0, len(current_blocks) - 1)
                    current_blocks.pop(rm_idx)
                    break
        
        return current_blocks

    # Main Loop: Multiple Restarts
    best_solution = None
    best_size = float('inf')
    
    # 1. Initial Greedy
    init_blocks = greedy_construct()
    improved = local_search(init_blocks, max_time=20)
    if len(improved) < best_size:
        best_size = len(improved)
        best_solution = improved[:]
        
    if time.time() > deadline - 30:
        return [list(all_blocks[idx]) for idx in best_solution]
        
    # 2. Iterated Local Search
    current_blocks = best_solution[:]
    
    while time.time() < deadline - 10:
        # Perturb: remove 5-10 random blocks
        n_remove = random.randint(5, 10)
        if len(current_blocks) > n_remove:
            to_remove = random.sample(range(len(current_blocks)), n_remove)
            current_blocks = [b for i, b in enumerate(current_blocks) if i not in to_remove]
            
        # Repair with Greedy
        covered = calculate_coverage(current_blocks)
        while not np.all(covered):
            if time.time() > deadline - 5:
                break
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            target_subset = random.choice(uncovered_indices.tolist())
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
        # Repair any gaps
        while not np.all(final_covered):
            if time.time() > deadline - 2:
                break
            uncovered = np.where(~final_covered)[0]
            if len(uncovered) == 0:
                break
            target = random.choice(uncovered.tolist())
            for b_idx in subset_to_blocks[target]:
                if b_idx not in best_solution:
                    best_solution.append(b_idx)
                    final_covered[block_covered_indices[b_idx]] = True
                    break
            else:
                # Shouldn't happen, but add random block
                best_solution.append(random.randint(0, n_blocks - 1))
                final_covered = calculate_coverage(best_solution)

    # Remove duplicates
    best_solution = list(dict.fromkeys(best_solution))

    return [list(all_blocks[idx]) for idx in best_solution]
