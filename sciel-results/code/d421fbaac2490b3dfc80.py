import itertools
import random
import time
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute for each block, the list of t-subset indices it covers
    # block_covered_indices[b_idx] = list of subset indices
    block_covered_indices = []
    # block_covered_set[b_idx] = set of subset indices for O(1) lookup
    block_covered_set = []
    
    for b in all_blocks:
        idxs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_covered_indices.append(idxs)
        block_covered_set.append(set(idxs))
        
    # For fast lookup of which blocks cover a specific t-subset
    # subset_to_blocks[s_idx] = list of block indices
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx, idxs in enumerate(block_covered_indices):
        for s_idx in idxs:
            subset_to_blocks[s_idx].append(b_idx)

    def calculate_coverage(block_indices):
        """Calculate coverage boolean array from a list of block indices."""
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
            # This is a heuristic to pick a "hard" subset to cover
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
        """Local search to reduce the number of blocks."""
        current_blocks = list(blocks)
        start_ls = time.time()
        
        # Maintain coverage count array for efficiency
        count_arr = np.zeros(n_subsets, dtype=int)
        for b_idx in current_blocks:
            count_arr[block_covered_indices[b_idx]] += 1
            
        improved = True
        while improved and time.time() < start_ls + max_time:
            improved = False
            
            # 1. Remove redundant blocks
            # A block is redundant if all its subsets are covered by other blocks
            # i.e., count_arr[s] > 1 for all s in block
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
                    improved = True
                    break # Restart loop to update counts
            
            if not improved and len(current_blocks) > 1:
                # 2. Try 2-for-1 exchange
                # Pick 2 random blocks
                if random.random() < 0.8: # Bias towards 2-for-1
                    i, j = random.sample(range(len(current_blocks)), 2)
                    b_i = current_blocks[i]
                    b_j = current_blocks[j]
                    
                    # Find critical subsets: subsets that would become uncovered if we remove b_i and b_j
                    # Critical if count_arr[s] == 1 and s is in b_i or b_j, OR count_arr[s] == 2 and s is in both
                    # Actually, if we remove b_i and b_j, a subset s is critical if:
                    # - s is in b_i and not in b_j, and count_arr[s] == 1
                    # - s is in b_j and not in b_i, and count_arr[s] == 1
                    # - s is in both, and count_arr[s] == 2
                    
                    set_i = block_covered_set[b_i]
                    set_j = block_covered_set[b_j]
                    
                    critical_subs = []
                    # Subsets in b_i
                    for s_idx in block_covered_indices[b_i]:
                        if s_idx in set_j:
                            if count_arr[s_idx] == 2:
                                critical_subs.append(s_idx)
                        else:
                            if count_arr[s_idx] == 1:
                                critical_subs.append(s_idx)
                    
                    # Subsets in b_j but not in b_i
                    for s_idx in block_covered_indices[b_j]:
                        if s_idx not in set_i:
                            if count_arr[s_idx] == 1:
                                critical_subs.append(s_idx)
                    
                    if not critical_subs:
                        # Can remove both without replacement
                        if i > j:
                            i, j = j, i
                        current_blocks.pop(j)
                        current_blocks.pop(i)
                        count_arr[block_covered_indices[b_i]] -= 1
                        count_arr[block_covered_indices[b_j]] -= 1
                        improved = True
                        break
                    
                    else:
                        # Need a block C that covers all critical_subs
                        # Find intersection of blocks covering each critical sub
                        # Start with blocks covering the first critical sub
                        candidate_blocks = set(subset_to_blocks[critical_subs[0]])
                        for s_idx in critical_subs[1:]:
                            candidate_blocks &= set(subset_to_blocks[s_idx])
                            if not candidate_blocks:
                                break
                        
                        if candidate_blocks:
                            # Pick a random candidate
                            best_c = random.choice(list(candidate_blocks))
                            # Ensure best_c is not b_i or b_j (it shouldn't be, as they are removed)
                            if best_c not in current_blocks:
                                # Remove i and j, add best_c
                                if i > j:
                                    i, j = j, i
                                current_blocks.pop(j)
                                current_blocks.pop(i)
                                current_blocks.append(best_c)
                                count_arr[block_covered_indices[b_i]] -= 1
                                count_arr[block_covered_indices[b_j]] -= 1
                                count_arr[block_covered_indices[best_c]] += 1
                                improved = True
                                break
                            else:
                                # best_c already in set, just remove i and j
                                if i > j:
                                    i, j = j, i
                                current_blocks.pop(j)
                                current_blocks.pop(i)
                                count_arr[block_covered_indices[b_i]] -= 1
                                count_arr[block_covered_indices[b_j]] -= 1
                                improved = True
                                break

        return current_blocks

    # Main Loop: Multiple Restarts / Iterated Local Search
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
        n_remove = random.randint(3, 7)
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
            # Pick a random uncovered subset
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
            # Accept with some probability to escape local optima (Simulated Annealing-like)
            # Or just keep the current best for ILS
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
