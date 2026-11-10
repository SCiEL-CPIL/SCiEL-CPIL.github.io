import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage: for each block, which t-subsets does it cover?
    block_covers = []
    for block in all_blocks:
        covered_indices = [subset_to_idx[t_sub] for t_sub in itertools.combinations(block, t)]
        block_covers.append(covered_indices)
    
    # Precompute: for each t-subset, which blocks cover it (for redundancy checks)
    t_subset_covered_by = [set() for _ in range(n_t_subsets)]
    for b_idx, covers in enumerate(block_covers):
        for t_idx in covers:
            t_subset_covered_by[t_idx].add(b_idx)
    
    def greedy_construct(seed_offset=0):
        """Greedy set cover with optional randomization."""
        coverage_count = [0] * n_t_subsets
        selected = []
        rng = random.Random(seed_offset)
        
        # Track which blocks are still candidates
        # We'll use a simple approach: at each step, score all blocks
        while True:
            if time.time() > deadline:
                break
            
            best_score = -1
            best_block_idx = -1
            
            # Count uncovered t-subsets
            uncovered_count = sum(1 for c in coverage_count if c == 0)
            if uncovered_count == 0:
                break
            
            # Score each block: primary = uncovered count, secondary = scarcity bonus
            # For speed, we only compute full score for blocks with high uncovered count
            # First pass: find max uncovered count
            max_uncov = 0
            for b_idx in range(n_blocks):
                unc = 0
                for t_idx in block_covers[b_idx]:
                    if coverage_count[t_idx] == 0:
                        unc += 1
                if unc > max_uncov:
                    max_uncov = unc
            
            if max_uncov == 0:
                break
            
            # Second pass: among blocks with max_uncov, pick best by scarcity
            best_block_idx = -1
            best_scarcity = -1
            
            for b_idx in range(n_blocks):
                unc = 0
                scarcity = 0.0
                for t_idx in block_covers[b_idx]:
                    if coverage_count[t_idx] == 0:
                        unc += 1
                        # Scarcity: how few other blocks also cover this t-subset
                        scarcity += 1.0 / max(1, len(t_subset_covered_by[t_idx]))
                
                if unc == max_uncov:
                    # Add small random perturbation for diversity
                    score = scarcity + rng.random() * 0.01
                    if best_block_idx == -1 or score > best_scarcity:
                        best_scarcity = score
                        best_block_idx = b_idx
            
            if best_block_idx == -1 or max_uncov == 0:
                break
            
            selected.append(best_block_idx)
            for t_idx in block_covers[best_block_idx]:
                coverage_count[t_idx] += 1
        
        return selected
    
    def local_search_deletion(solution_indices, time_limit):
        """Remove redundant blocks iteratively."""
        coverage_count = [0] * n_t_subsets
        for b_idx in solution_indices:
            for t_idx in block_covers[b_idx]:
                coverage_count[t_idx] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(solution_indices)):
                if time.time() > time_limit:
                    break
                
                b_idx = solution_indices[i]
                
                # Check if removal is valid: no t-subset has count 1
                can_remove = True
                for t_idx in block_covers[b_idx]:
                    if coverage_count[t_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for t_idx in block_covers[b_idx]:
                        coverage_count[t_idx] -= 1
                    solution_indices.pop(i)
                    improved = True
                    break
        
        return solution_indices
    
    def local_search_swap(solution_indices, time_limit):
        """Try to replace one block with another that covers more uncovered t-subsets."""
        coverage_count = [0] * n_t_subsets
        for b_idx in solution_indices:
            for t_idx in block_covers[b_idx]:
                coverage_count[t_idx] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Find t-subsets with minimum coverage
            min_cov = min(coverage_count)
            if min_cov >= 2:
                # Try to find a block that covers many min-coverage t-subsets
                # and replace a redundant block
                pass
            
            # Try removing any block and see if we can add a better one
            for i in range(len(solution_indices)):
                if time.time() > time_limit:
                    break
                
                b_idx = solution_indices[i]
                
                # Tentatively remove
                temp_cov = list(coverage_count)
                for t_idx in block_covers[b_idx]:
                    temp_cov[t_idx] -= 1
                
                # Check if still valid
                if any(c == 0 for c in temp_cov):
                    continue
                
                # Find if there's a block that covers more of the now-scarce t-subsets
                # and would allow further reduction
                # This is complex; for now, just try simple deletions
                
                # Actually, let's try a different approach: 
                # For each pair (i, j) where i != j, try swapping block i with block j
                # if it reduces total coverage redundancy
                # This is O(n^2) which might be too slow for ~150 blocks
                
                # Simpler: just try to find a block that covers the most 
                # t-subsets with coverage == 1, and if it's not in the solution,
                # consider adding it and then removing a redundant block.
        
        return solution_indices
    
    # Phase 1: Try multiple greedy constructions with different random seeds
    best_solution = None
    best_size = float('inf')
    
    # Allocate ~60% of time to construction phase
    construction_deadline = start_time + (deadline - start_time) * 0.55
    
    # First try deterministic greedy (seed 0)
    sol = greedy_construct(seed_offset=0)
    if len(sol) < best_size:
        best_size = len(sol)
        best_solution = sol[:]
    
    # Then try randomized versions
    seed = 1
    while time.time() < construction_deadline:
        sol = greedy_construct(seed_offset=seed)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
            # If we found a significantly better solution, reset counter
        seed += 1
        if seed % 10 == 0 and time.time() > construction_deadline:
            break
    
    # Phase 2: Local search refinement on best solution
    refinement_deadline = deadline - 5.0  # Leave 5s safety margin
    
    # First pass: simple deletion
    best_solution = local_search_deletion(best_solution, min(refinement_deadline, start_time + (deadline - start_time) * 0.85))
    
    # Second pass: try more aggressive local search with swaps
    # Try to identify blocks that are "almost" redundant and see if swapping helps
    coverage_count = [0] * n_t_subsets
    for b_idx in best_solution:
        for t_idx in block_covers[b_idx]:
            coverage_count[t_idx] += 1
    
    # Find blocks that cover the most unique t-subsets (coverage == 1)
    # These are critical blocks. Blocks that don't cover any unique t-subsets are candidates for removal.
    
    # Try a more sophisticated local search: 
    # For each block, check if removing it and adding one or two other blocks 
    # results in a smaller or equal size solution with better coverage properties.
    
    # Since time is limited, focus on deletion-based local search with multiple passes
    for pass_num in range(5):
        if time.time() > refinement_deadline:
            break
        best_solution = local_search_deletion(best_solution, refinement_deadline)
        if len(best_solution) == best_size:
            break
    
    return [all_blocks[i] for i in best_solution]
