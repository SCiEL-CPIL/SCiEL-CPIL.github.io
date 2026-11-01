import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 108.0  # 12s safety margin before 120s limit
    
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # For each block, precompute its covered subsets as a Python int bitmask
    # Each bit position corresponds to a 4-subset index
    block_bitmasks = []
    block_to_subsets = []  # Still needed for local search
    for b in all_blocks:
        subs = []
        mask = 0
        for s in itertools.combinations(b, t):
            idx = subset_to_idx[s]
            subs.append(idx)
            mask |= (1 << idx)
        block_bitmasks.append(mask)
        block_to_subsets.append(subs)
    
    # For each subset, list of block indices that cover it (for local search)
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)
    
    full_mask = (1 << num_subsets) - 1
    
    best_solution = None
    best_len = float('inf')
    
    def greedy_bitset():
        """Bitset-vectorized greedy set cover with randomized tie-breaking."""
        uncovered = full_mask
        solution = []
        
        while uncovered:
            if time.time() > deadline:
                return solution if len(solution) < best_len else None
            
            # Find the block with maximum popcount of (block_mask & uncovered)
            max_count = 0
            candidates = []
            for b_idx in range(num_blocks):
                count = (block_bitmasks[b_idx] & uncovered).bit_count()
                if count > max_count:
                    max_count = count
                    candidates = [b_idx]
                elif count == max_count and count > 0:
                    candidates.append(b_idx)
            
            if not candidates:
                break
            
            best_b = random.choice(candidates)
            solution.append(best_b)
            uncovered &= ~block_bitmasks[best_b]
        
        return solution

    def local_search(solution):
        """Local search with removal and swap operations."""
        coverage = [0] * num_subsets
        for b in solution:
            for s in block_to_subsets[b]:
                coverage[s] += 1
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            n = len(solution)
            for i in range(n):
                if time.time() > deadline:
                    break
                    
                b = solution[i]
                # Check if redundant
                critical_subs = []
                for s in block_to_subsets[b]:
                    if coverage[s] == 1:
                        critical_subs.append(s)
                        
                if not critical_subs:
                    # Redundant, remove
                    for s in block_to_subsets[b]:
                        coverage[s] -= 1
                    solution.pop(i)
                    improved = True
                    break
                    
                else:
                    # Critical, try to find a replacement
                    if len(critical_subs) > 0:
                        # Start with the first critical sub's blocks
                        candidates = set(subset_to_blocks[critical_subs[0]])
                        for s in critical_subs[1:]:
                            candidates.intersection_update(subset_to_blocks[s])
                            if not candidates:
                                break
                                
                        if candidates:
                            # Filter out the current block and any block already in solution
                            sol_set = set(solution)
                            valid_candidates = [c for c in candidates if c not in sol_set]
                            
                            if valid_candidates:
                                # Pick one that maximizes additional coverage
                                best_new_b = None
                                best_additional = -1
                                for new_b in valid_candidates:
                                    additional = 0
                                    for s in block_to_subsets[new_b]:
                                        if coverage[s] == 1:
                                            additional += 1
                                    if additional > best_additional:
                                        best_additional = additional
                                        best_new_b = new_b
                                        
                                if best_new_b is not None and best_additional > 0:
                                    for s in block_to_subsets[b]:
                                        coverage[s] -= 1
                                    for s in block_to_subsets[best_new_b]:
                                        coverage[s] += 1
                                        
                                    solution[i] = best_new_b
                                    improved = True
                                    break
                                
        return solution

    def constrained_swap_search(solution):
        """Try 1-for-1 and 1-for-2 swaps on the best solution."""
        coverage = [0] * num_subsets
        for b in solution:
            for s in block_to_subsets[b]:
                coverage[s] += 1
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            n = len(solution)
            for i in range(n):
                if time.time() > deadline:
                    break
                    
                b = solution[i]
                critical_subs = []
                for s in block_to_subsets[b]:
                    if coverage[s] == 1:
                        critical_subs.append(s)
                        
                if not critical_subs:
                    for s in block_to_subsets[b]:
                        coverage[s] -= 1
                    solution.pop(i)
                    improved = True
                    break
                else:
                    if len(critical_subs) <= 2:
                        candidate_blocks = set()
                        for s in critical_subs:
                            candidate_blocks.update(subset_to_blocks[s])
                        candidate_blocks.discard(b)
                        candidate_blocks -= set(solution)
                        
                        candidates = list(candidate_blocks)
                        random.shuffle(candidates)
                        limit = min(len(candidates), 60)
                        
                        # Try 1-for-1 first
                        for j in range(limit):
                            new_b = candidates[j]
                            # Check if new_b covers all critical subsets
                            if all(s in block_to_subsets[new_b] for s in critical_subs):
                                for s in block_to_subsets[b]:
                                    coverage[s] -= 1
                                for s in block_to_subsets[new_b]:
                                    coverage[s] += 1
                                solution[i] = new_b
                                improved = True
                                break
                        
                        if improved:
                            continue
                        
                        # Try 1-for-2
                        if len(critical_subs) <= 3:
                            for j in range(limit):
                                for l in range(j+1, limit):
                                    b1, b2 = candidates[j], candidates[l]
                                    covered = set()
                                    for s in block_to_subsets[b1]:
                                        if s in critical_subs:
                                            covered.add(s)
                                    for s in block_to_subsets[b2]:
                                        if s in critical_subs:
                                            covered.add(s)
                                    
                                    if all(cs in covered for cs in critical_subs):
                                        for s in block_to_subsets[b]:
                                            coverage[s] -= 1
                                        for s in block_to_subsets[b1]:
                                            coverage[s] += 1
                                        for s in block_to_subsets[b2]:
                                            coverage[s] += 1
                                        
                                        solution.pop(i)
                                        solution.extend([b1, b2])
                                        improved = True
                                        break
                                if improved:
                                    break
                        
        return solution

    # Run multiple bitset-greedy starts with local search
    num_runs = 40
    for run in range(num_runs):
        if time.time() > deadline:
            break
        
        sol = greedy_bitset()
        if sol is None:
            continue
        
        sol = local_search(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol.copy()
    
    # Try additional constrained swaps on best solution
    if best_solution is not None and time.time() < deadline:
        sol = constrained_swap_search(best_solution.copy())
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol.copy()
    
    if best_solution is None:
        sol = greedy_bitset()
        return [list(all_blocks[b]) for b in sol]
        
    return [list(all_blocks[b]) for b in best_solution]

if __name__ == "__main__":
    blocks = construct(19, 7, 4)
    print(len(blocks))
