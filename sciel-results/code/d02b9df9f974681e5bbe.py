import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute block_subsets as a numpy boolean matrix for fast vectorized operations
    # Coverage matrix: shape (n_blocks, n_t_subsets)
    # M[i, j] = 1 if block i covers t-subset j
    # Memory: 50388 * 3876 * 1 byte (bool) ~ 195 MB. Acceptable.
    coverage_matrix = np.zeros((n_blocks, n_t_subsets), dtype=bool)
    
    # Also keep lists for iteration when needed
    block_subsets = [[] for _ in range(n_blocks)]
    # subset_to_blocks[j] = list of block indices covering t-subset j
    subset_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            coverage_matrix[i, idx] = True
            block_subsets[i].append(idx)
            subset_to_blocks[idx].append(i)
            
    def greedy_construct():
        """
        Vectorized greedy construction.
        Returns a list of block indices.
        """
        in_block = np.zeros(n_blocks, dtype=bool)
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        uncovered_count = n_t_subsets
        solution = []
        
        while uncovered_count > 0:
            # Calculate gain for all blocks not yet selected
            # gain[i] = number of uncovered subsets covered by block i
            # uncovered = (cover_count == 0)
            uncovered_mask = (cover_count == 0)
            
            # gain = sum(coverage_matrix[i] & uncovered_mask) for each i
            # Using matrix multiplication or sum along axis
            # gain = np.sum(coverage_matrix & uncovered_mask, axis=1)
            # But we need to ignore blocks already in solution (gain should be 0 or -1)
            gains = np.sum(coverage_matrix & uncovered_mask, axis=1)
            gains[in_block] = -1
            
            best_block = np.argmax(gains)
            max_gain = gains[best_block]
            
            if max_gain <= 0:
                # Should not happen if problem is solvable
                break
                
            # Add block
            in_block[best_block] = True
            solution.append(best_block)
            
            # Update cover counts
            # New covers = coverage_matrix[best_block] & uncovered_mask
            new_covers = coverage_matrix[best_block] & uncovered_mask
            cover_count += new_covers.astype(np.int32)
            uncovered_count -= np.sum(new_covers)
            
        return solution

    def remove_redundants(solution, cover_count):
        """
        Remove redundant blocks from solution.
        Returns updated solution and cover_count.
        """
        # Re-initialize in_block from solution
        in_block = np.zeros(n_blocks, dtype=bool)
        for b in solution:
            in_block[b] = True
            
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of solution
            for b in list(solution):
                # Check if b is redundant: all subsets in b have cover_count > 1
                # i.e., min(cover_count[subset] for subset in block_subsets[b]) > 1
                # Or: sum(cover_count[subset] == 1 for subset in block_subsets[b]) == 0
                # Using numpy:
                # criticals = np.sum(cover_count[block_subsets[b]] == 1)
                # But block_subsets[b] is a list.
                subs = block_subsets[b]
                if len(subs) == 0:
                    continue
                # Check if any subset in b is covered only once
                # If yes, b is critical (cannot be removed)
                # If no, b is redundant
                # counts = cover_count[subs]
                # if np.min(counts) > 1:
                #     remove b
                # Optimization: just check sum == 0
                # Actually, if cover_count[s] == 1 for any s in b, we cannot remove b.
                # So we check if np.any(cover_count[subs] == 1)
                # If not, remove.
                
                # Faster check:
                # counts = cover_count[subs]
                # if np.all(counts > 1):
                #     remove
                # Let's use a loop for small subs (35 elements) to avoid numpy overhead?
                # 35 elements is small. Python loop might be faster than numpy indexing.
                is_critical = False
                for s_idx in subs:
                    if cover_count[s_idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    # Remove b
                    in_block[b] = False
                    solution.remove(b)
                    for s_idx in subs:
                        cover_count[s_idx] -= 1
                    changed = True
                    break # Restart loop
                    
        return solution, cover_count

    best_solution = None
    best_count = float('inf')
    
    # Initial Greedy
    sol = greedy_construct()
    # Compute cover_count for this solution
    cover_count = np.zeros(n_t_subsets, dtype=np.int32)
    for b in sol:
        cover_count += coverage_matrix[b].astype(np.int32)
        
    sol, cover_count = remove_redundants(sol, cover_count)
    
    if len(sol) < best_count:
        best_count = len(sol)
        best_solution = list(sol)
        
    # Local Search / Simulated Annealing
    # We will try to swap blocks to reduce size or improve "slack"
    # Objective: Minimize len(solution)
    # Secondary: Maximize slack (sum of (cover_count - 1) for all subsets)
    
    current_solution = list(best_solution)
    current_cover_count = cover_count.copy()
    current_len = len(current_solution)
    
    # Precompute block_subsets as numpy arrays for faster access?
    # block_subsets_np = [np.array(s) for s in block_subsets]
    
    # Temperature for simulated annealing
    T = 1.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if not current_solution:
            break
            
        # Pick a random block to remove
        b_out = random.choice(current_solution)
        
        # Find critical subsets for b_out
        # Criticals are subsets in b_out with cover_count == 1
        criticals = []
        for s_idx in block_subsets[b_out]:
            if current_cover_count[s_idx] == 1:
                criticals.append(s_idx)
                
        if not criticals:
            # b_out is redundant, remove it
            current_solution.remove(b_out)
            for s_idx in block_subsets[b_out]:
                current_cover_count[s_idx] -= 1
            current_len -= 1
            # Remove other redundants
            current_solution, current_cover_count = remove_redundants(current_solution, current_cover_count)
            current_len = len(current_solution)
            
            if current_len < best_count:
                best_count = current_len
                best_solution = list(current_solution)
            continue
            
        # Find candidate b_in that covers all criticals
        # Candidates are blocks that cover criticals[0]
        candidates = subset_to_blocks[criticals[0]]
        
        best_candidate = -1
        best_score = -float('inf')
        
        # For each candidate, check if it covers all other criticals
        # If yes, calculate "gain" or "cost"
        # Cost: number of subsets that become over-covered?
        # Or simply: prefer candidates that cover many *uncovered* subsets? 
        # But all subsets are covered.
        # Prefer candidates that have high "redundancy" in other areas?
        # Let's just accept the first valid one found, or one with max overlap with current solution?
        # No, we want to swap.
        # Let's score candidates by how many *other* subsets they cover that are currently covered exactly once?
        # No, that would increase criticals.
        
        # Simple heuristic: Pick a random valid candidate.
        # Or pick the one with max gain in terms of "slack"?
        
        # Let's iterate candidates
        found_valid = False
        for b_in in candidates:
            if b_in in current_solution:
                continue
                
            # Check if b_in covers all criticals
            valid = True
            for s_idx in criticals[1:]:
                if not coverage_matrix[b_in, s_idx]:
                    valid = False
                    break
                    
            if valid:
                # Calculate change in objective
                # Remove b_out, add b_in
                # New cover counts:
                # cover_count_new = cover_count_old - cover(b_out) + cover(b_in)
                
                # Check if this swap is valid (it is by construction of criticals)
                # Does it reduce size? No, size stays same.
                # Does it improve slack?
                # Slack = sum(cover_count - 1)
                # Change in slack = sum(cover(b_in) - 1) - sum(cover(b_out) - 1)
                # = sum(cover(b_in)) - sum(cover(b_out))
                # Since both cover C(7,4)=35 subsets, this is 0.
                # So slack doesn't change directly.
                
                # However, the *distribution* of cover counts changes.
                # We want to move towards a state where more blocks are redundant.
                # A good heuristic: Maximize the number of subsets that become covered >= 2 times
                # by b_in that were previously covered exactly once by b_out?
                
                # Let's just accept the swap if it doesn't increase the number of "critical" blocks?
                # Or use simulated annealing on "potential" = sum(1 / cover_count[s])?
                
                # Let's try: Accept if random() < exp(-delta_E / T)
                # Let delta_E be change in "energy" where energy = sum(cover_count[s] == 1)
                # We want to minimize the number of critical subsets.
                
                # Count criticals in b_out (we already have this: len(criticals))
                # Count criticals that will be created by b_in?
                # Subsets covered by b_in that are currently covered exactly once (and not in b_out)
                # Subsets covered by b_out that are currently covered exactly once (and not in b_in) will lose a cover.
                
                # Let's compute new critical count roughly.
                # This is expensive.
                
                # Simpler: Just accept the first valid candidate.
                found_valid = True
                best_candidate = b_in
                break
                
        if found_valid:
            # Perform swap
            current_solution.remove(b_out)
            for s_idx in block_subsets[b_out]:
                current_cover_count[s_idx] -= 1
                
            current_solution.append(best_candidate)
            for s_idx in block_subsets[best_candidate]:
                current_cover_count[s_idx] += 1
                
            # Remove redundants
            current_solution, current_cover_count = remove_redundants(current_solution, current_cover_count)
            current_len = len(current_solution)
            
            if current_len < best_count:
                best_count = current_len
                best_solution = list(current_solution)
                
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    # Final polish: try to remove redundants again
    best_solution, _ = remove_redundants(best_solution, cover_count)
    
    return [list(all_blocks[i]) for i in best_solution]
