import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 108.0
    
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block properties
    block_bitmasks = []
    block_to_subsets = []
    block_sizes = []  # Number of unique t-subsets per block
    for b in all_blocks:
        subs = []
        mask = 0
        for s in itertools.combinations(b, t):
            idx = subset_to_idx[s]
            subs.append(idx)
            mask |= (1 << idx)
        block_bitmasks.append(mask)
        block_to_subsets.append(subs)
        block_sizes.append(len(subs))
    
    # Collision graph: for each pair of blocks, count shared t-subsets
    # This is expensive, so we'll compute it lazily or use a sparse representation
    # For now, precompute overlap counts using numpy for efficiency
    # Convert bitmasks to numpy arrays for fast operations
    # But bitmasks are Python ints, so let's use a different approach
    
    # Build subset-to-blocks mapping for quick lookup
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx, subs in enumerate(block_to_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)
    
    full_mask = (1 << num_subsets) - 1
    
    best_solution = None
    best_len = float('inf')
    
    def compute_collision_score(block_idx, selected_blocks):
        """Compute how many t-subsets of block_idx are already covered by selected blocks.
        Lower score = less collision = better for group testing diversity."""
        if not selected_blocks:
            return 0
        # Count how many t-subsets of this block are already in selected blocks
        collisions = 0
        for s_idx in block_to_subsets[block_idx]:
            # Check if any selected block covers this subset
            for b in selected_blocks:
                if s_idx in block_to_subsets[b]:
                    collisions += 1
                    break
        return collisions
    
    def greedy_group_testing():
        """Greedy set cover with group testing: prefer blocks with minimal collision
        with already-selected blocks to maximize diversity."""
        uncovered = full_mask
        solution = []
        selected_set = set()
        
        while uncovered:
            if time.time() > deadline:
                return solution if len(solution) < best_len else None
            
            # For each candidate block, compute:
            # 1. Coverage: how many currently uncovered t-subsets it covers
            # 2. Collision: how many of its t-subsets are already in selected blocks
            # Score = coverage - alpha * collision (we want high coverage, low collision)
            
            max_score = -1
            best_candidates = []
            
            for b_idx in range(num_blocks):
                # Coverage of uncovered subsets
                cov_mask = block_bitmasks[b_idx] & uncovered
                coverage = cov_mask.bit_count()
                
                if coverage == 0:
                    continue
                
                # Collision: count how many t-subsets of this block are already covered
                collisions = 0
                for s_idx in block_to_subsets[b_idx]:
                    # Check if this subset is covered by any selected block
                    for sb in selected_set:
                        if s_idx in block_to_subsets[sb]:
                            collisions += 1
                            break
                
                # Score: prioritize coverage, penalize collisions
                # alpha controls the trade-off; start with 0.1
                alpha = 0.1
                score = coverage - alpha * collisions
                
                if score > max_score + 1e-9:
                    max_score = score
                    best_candidates = [b_idx]
                elif abs(score - max_score) < 1e-9:
                    best_candidates.append(b_idx)
            
            if not best_candidates:
                break
            
            # Random tie-breaking for diversity
            best_b = random.choice(best_candidates)
            solution.append(best_b)
            selected_set.add(best_b)
            uncovered &= ~block_bitmasks[best_b]
        
        return solution
    
    def greedy_bitset():
        """Original bitset-vectorized greedy set cover with randomized tie-breaking."""
        uncovered = full_mask
        solution = []
        
        while uncovered:
            if time.time() > deadline:
                return solution if len(solution) < best_len else None
            
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
                    if len(critical_subs) > 0:
                        candidates = set(subset_to_blocks[critical_subs[0]])
                        for s in critical_subs[1:]:
                            candidates.intersection_update(subset_to_blocks[s])
                            if not candidates:
                                break
                                
                        if candidates:
                            sol_set = set(solution)
                            valid_candidates = [c for c in candidates if c not in sol_set]
                            
                            if valid_candidates:
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
                        
                        for j in range(limit):
                            new_b = candidates[j]
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
    
    # Run multiple strategies: group testing, standard greedy, and hybrid
    num_runs = 30
    for run in range(num_runs):
        if time.time() > deadline:
            break
        
        # Alternate between strategies
        if run % 2 == 0:
            sol = greedy_group_testing()
        else:
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
