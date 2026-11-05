import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)
        
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)

    # --- Phase 1: Initial Greedy Construction ---
    # uncovered[s] = 1 if t-subset s is not covered
    uncovered = np.ones(n_t_subs, dtype=np.int32)
    total_uncovered = n_t_subs
    solution_blocks = []
    solution_set = set()
    
    # uncovered_count_per_block[b] = number of t-subsets in block b that are currently uncovered
    uncovered_count_per_block = np.array([len(block_to_tsubs[b]) for b in range(n_blocks)], dtype=np.int32)
    
    while total_uncovered > 0 and time.time() < deadline:
        # Find block with max uncovered count
        max_count = np.max(uncovered_count_per_block)
        if max_count == 0:
            break
            
        # Pick one of the blocks with max count (random tie-breaking for diversity)
        candidates = np.where(uncovered_count_per_block == max_count)[0]
        valid_block_idx = random.choice(candidates)
        
        solution_blocks.append(valid_block_idx)
        solution_set.add(valid_block_idx)
        
        # Update uncovered status and counts
        for s in block_to_tsubs[valid_block_idx]:
            if uncovered[s]:
                uncovered[s] = 0
                total_uncovered -= 1
                # Decrease count for all blocks covering s
                for b_idx in tsub_to_blocks[s]:
                    uncovered_count_per_block[b_idx] -= 1

    # --- Phase 2: Simulated Annealing ---
    # cover_count[s] = number of blocks in solution covering t-subset s
    cover_count = np.zeros(n_t_subs, dtype=np.int32)
    for b_idx in solution_blocks:
        cover_count[block_to_tsubs[b_idx]] += 1
            
    best_solution = list(solution_blocks)
    best_len = len(solution_blocks)
    
    # Helper to check if a solution is valid
    def is_valid(sol):
        if not sol: return n_t_subs == 0
        cc = np.zeros(n_t_subs, dtype=np.int32)
        for b in sol:
            cc[block_to_tsubs[b]] += 1
        return np.all(cc > 0)

    # Ensure initial solution is valid
    if not is_valid(best_solution):
         for s in range(n_t_subs):
             if cover_count[s] == 0:
                 b = tsub_to_blocks[s][0]
                 if b not in solution_set:
                     solution_blocks.append(b)
                     solution_set.add(b)
                     cover_count[block_to_tsubs[b]] += 1
         best_solution = list(solution_blocks)
         best_len = len(solution_blocks)

    # Simulated Annealing parameters
    T = 100.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    max_iter = 50000
    
    current_solution = list(solution_blocks)
    current_cover_count = cover_count.copy()
    current_set = set(solution_blocks)
    
    for iter_num in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Try to remove a redundant block
        n_sol = len(current_solution)
        if n_sol > 0:
            # Find redundant blocks (all their t-subsets have cover_count > 1)
            redundant_candidates = []
            for i, b_idx in enumerate(current_solution):
                if np.all(current_cover_count[block_to_tsubs[b_idx]] > 1):
                    redundant_candidates.append(i)
            
            if redundant_candidates:
                # Remove a random redundant block
                idx = random.choice(redundant_candidates)
                b_idx = current_solution[idx]
                current_cover_count[block_to_tsubs[b_idx]] -= 1
                current_solution.pop(idx)
                current_set.remove(b_idx)
                
                if len(current_solution) < best_len:
                    best_len = len(current_solution)
                    best_solution = list(current_solution)
                continue
        
        # Try swap: remove one block, add another
        if n_sol > 0 and time.time() < deadline - 5:
            # Pick a random block to remove
            rem_idx = random.randint(0, n_sol - 1)
            rem_b = current_solution[rem_idx]
            
            # Find critical subsets (those with cover_count == 1)
            crit_subs = [s for s in block_to_tsubs[rem_b] if current_cover_count[s] == 1]
            
            if not crit_subs:
                # No critical subsets, can safely remove and add any block
                # This is equivalent to a pure removal, but we'll do swap for diversity
                pass
            
            # Find candidate blocks that cover all critical subsets
            if crit_subs:
                candidate_set = set(tsub_to_blocks[crit_subs[0]])
                for s in crit_subs[1:]:
                    candidate_set.intersection_update(tsub_to_blocks[s])
                    if not candidate_set:
                        candidate_set = set()
                        break
            else:
                candidate_set = set(range(n_blocks))
            
            # Filter out blocks already in solution
            candidates = [c for c in candidate_set if c not in current_set]
            
            if candidates:
                # Pick a random candidate
                add_b = random.choice(candidates)
                
                # Calculate energy change (not used directly, but for acceptance)
                # In SA, we accept based on temperature
                # For now, always accept improving moves, sometimes accept worse
                
                # Perform swap
                current_cover_count[block_to_tsubs[rem_b]] -= 1
                current_cover_count[block_to_tsubs[add_b]] += 1
                
                current_solution[rem_idx] = add_b
                current_set.remove(rem_b)
                current_set.add(add_b)
                
                if len(current_solution) < best_len:
                    best_len = len(current_solution)
                    best_solution = list(current_solution)
            
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min

    # Final Polish: Remove redundant blocks from best_solution
    if best_solution:
        final_cc = np.zeros(n_t_subs, dtype=np.int32)
        for b in best_solution:
            final_cc[block_to_tsubs[b]] += 1
        
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for i in range(len(best_solution)):
                b = best_solution[i]
                if np.all(final_cc[block_to_tsubs[b]] > 1):
                    final_cc[block_to_tsubs[b]] -= 1
                    best_solution.pop(i)
                    changed = True
                    break

    # Final Validation
    if not is_valid(best_solution):
        final_cc = np.zeros(n_t_subs, dtype=np.int32)
        for b in best_solution:
            final_cc[block_to_tsubs[b]] += 1
        for s in range(n_t_subs):
            if final_cc[s] == 0:
                b = tsub_to_blocks[s][0]
                if b not in set(best_solution):
                    best_solution.append(b)
                    final_cc[block_to_tsubs[b]] += 1

    return [list(all_blocks[b_idx]) for b_idx in best_solution]
