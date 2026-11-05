import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precomputation
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = tris
        
    # Convert to numpy arrays for vectorized operations
    max_tris = 35  # C(7,3)
    block_triple_matrix = np.full((num_blocks, max_tris), -1, dtype=np.int32)
    block_triple_lens = np.zeros(num_blocks, dtype=np.int32)
    
    for i, tris in enumerate(block_triple_indices):
        block_triple_matrix[i, :len(tris)] = tris
        block_triple_lens[i] = len(tris)
    
    # Precompute triple_to_blocks as numpy arrays for faster intersection
    triple_to_blocks_arr = [np.array(b, dtype=np.int32) for b in triple_to_blocks]
    
    best_sol = None
    best_energy = float('inf')
    
    seed_counter = 0
    restarts = 0
    
    while time.time() < deadline - 12:
        seed_counter += 1
        restarts += 1
        rng = random.Random(seed_counter)
        
        # 1. Fast Greedy Construction with randomized tie-breaking
        sol = _fast_greedy(rng, num_blocks, block_triple_matrix, block_triple_lens, num_triples, deadline - 10)
        
        # 2. Remove Redundant Blocks
        sol = _remove_redundant(sol, block_triple_matrix, block_triple_lens, num_triples)
        
        # 3. Local Search (SA) with adaptive temperature
        time_left_for_sa = deadline - 10 - time.time()
        if time_left_for_sa < 1:
            break
            
        sol = _local_search_v2(sol, num_blocks, block_triple_matrix, block_triple_lens, 
                               triple_to_blocks_arr, num_triples, time_left_for_sa, rng)
        
        # 4. Final polish: deterministic removal + targeted swap
        sol = _polish(sol, num_blocks, block_triple_matrix, block_triple_lens, 
                      triple_to_blocks_arr, num_triples)
        
        # Evaluate
        if sol:
            covered = np.zeros(num_triples, dtype=bool)
            for b_idx in sol:
                tris = block_triple_matrix[b_idx]
                valid_tris = tris[tris != -1]
                if len(valid_tris) > 0:
                    covered[valid_tris] = True
            
            uncovered_count = np.sum(~covered)
            
            if uncovered_count == 0:
                energy = len(sol)
                if energy < best_energy:
                    best_energy = energy
                    best_sol = list(sol)

    if best_sol is None:
        return []
        
    # Convert indices back to point lists
    return [list(all_blocks[i]) for i in best_sol]

def _fast_greedy(rng, num_blocks, block_triple_matrix, block_triple_lens, num_triples, deadline):
    """Fast greedy with randomized tie-breaking and early termination."""
    uncovered = np.ones(num_triples, dtype=bool)
    selected = []
    selected_set = set()
    
    # Precompute mask and safe matrix once
    mask = block_triple_matrix != -1
    safe_matrix = np.where(mask, block_triple_matrix, 0)
    
    while np.any(uncovered) and time.time() < deadline:
        # Calculate gain for all blocks
        vals = uncovered[safe_matrix]
        vals = np.where(mask, vals, 0)
        gains = vals.sum(axis=1)
        
        # Exclude already selected blocks
        if selected_set:
            gains[list(selected_set)] = -1
        
        best_gain = np.max(gains)
        
        if best_gain <= 0:
            break
        
        # Randomized selection among top candidates
        candidates = np.where(gains == best_gain)[0]
        if len(candidates) > 1:
            # Pick randomly from top candidates for diversity
            chosen = int(rng.choice(candidates))
        else:
            chosen = int(candidates[0])
        
        selected.append(chosen)
        selected_set.add(chosen)
        
        # Update uncovered
        tris = block_triple_matrix[chosen]
        valid_tris = tris[tris != -1]
        uncovered[valid_tris] = False
        
    return selected

def _remove_redundant(sol, block_triple_matrix, block_triple_lens, num_triples):
    if not sol:
        return sol
        
    # Count coverage
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1
            
    sol_set = set(sol)
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            # Check if removing idx leaves any triple with count 0
            tris = block_triple_matrix[idx]
            valid_tris = tris[tris != -1]
            if len(valid_tris) == 0:
                continue
            if np.any(counts[valid_tris] == 1):
                continue
            # Redundant
            counts[valid_tris] -= 1
            sol.remove(idx)
            sol_set.discard(idx)
            changed = True
            
    return sol

def _local_search_v2(initial_sol, num_blocks, block_triple_matrix, block_triple_lens, 
                     triple_to_blocks_arr, num_triples, time_limit, rng):
    """Improved local search with adaptive energy and better swap logic."""
    sol = list(initial_sol)
    sol_set = set(sol)
    
    # Coverage counts
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1
            
    current_uncovered = int(np.sum(counts == 0))
    
    def get_energy():
        # Adaptive weight: higher when many uncovered, lower when few
        w = 50 + 50 * current_uncovered if current_uncovered > 0 else 1
        return len(sol) + w * current_uncovered
    
    current_energy = get_energy()
    best_sol = list(sol)
    best_energy = current_energy
    
    T = 8.0
    cooling = 0.997
    min_T = 0.05
    
    while time.time() < time_limit and T > min_T:
        move_type = rng.random()
        
        if move_type < 0.5 and sol:
            # Removal
            to_remove = rng.choice(sol)
            tris = block_triple_matrix[to_remove]
            valid_tris = tris[tris != -1]
            
            if len(valid_tris) == 0:
                continue
            
            min_count_in_block = int(np.min(counts[valid_tris]))
            
            if min_count_in_block > 1:
                # Redundant, always accept
                counts[valid_tris] -= 1
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                current_uncovered = int(np.sum(counts == 0))
                current_energy = get_energy()
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)
            else:
                delta_uncovered = int(np.sum(counts[valid_tris] == 1))
                w = 50 + 50 * (current_uncovered + delta_uncovered) if (current_uncovered + delta_uncovered) > 0 else 1
                delta_e = 1 + w * delta_uncovered - (50 + 50 * current_uncovered if current_uncovered > 0 else 1) * 0
                # Simpler: just compute new energy directly
                new_uncovered = current_uncovered + delta_uncovered
                new_w = 50 + 50 * new_uncovered if new_uncovered > 0 else 1
                new_energy = (len(sol) - 1) + new_w * new_uncovered
                delta_e = new_energy - current_energy
                
                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    counts[valid_tris] -= 1
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)

        elif move_type < 0.85 and sol:
            # Swap: remove one block, add another that covers some of the lost triples
            to_remove = rng.choice(sol)
            tris_remove = block_triple_matrix[to_remove]
            valid_tris_remove = tris_remove[tris_remove != -1]
            
            if len(valid_tris_remove) == 0:
                continue
            
            # Find triples that will become uncovered (count == 1)
            uncovered_mask = (counts[valid_tris_remove] == 1)
            uncovered_tris = valid_tris_remove[uncovered_mask]
            
            if len(uncovered_tris) == 0:
                # Pure removal case
                continue
            
            # Find candidate blocks that cover at least one uncovered triple
            # and are not already in solution
            candidate_blocks = set()
            for t_idx in uncovered_tris:
                candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
            
            candidate_blocks.difference_update(sol_set)
            
            if not candidate_blocks:
                continue
            
            # Score candidates by how many uncovered_tris they cover
            best_candidate = None
            best_score = -1
            candidates_list = list(candidate_blocks)
            
            # If too many candidates, sample
            if len(candidates_list) > 50:
                candidates_list = rng.sample(candidates_list, 50)
            
            for cand in candidates_list:
                tris_add = block_triple_matrix[cand]
                valid_tris_add = tris_add[tris_add != -1]
                # Count how many of uncovered_tris are in valid_tris_add
                score = int(np.sum(np.isin(valid_tris_add, uncovered_tris)))
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            if best_candidate is None or best_score == 0:
                continue
            
            to_add = best_candidate
            tris_add = block_triple_matrix[to_add]
            valid_tris_add = tris_add[tris_add != -1]
            
            # Compute new uncovered
            new_counts = counts.copy()
            new_counts[valid_tris_remove] -= 1
            new_counts[valid_tris_add] += 1
            new_uncovered = int(np.sum(new_counts == 0))
            
            new_w = 50 + 50 * new_uncovered if new_uncovered > 0 else 1
            new_energy = len(sol) + new_w * new_uncovered  # same length, so len(sol)
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                # Accept
                counts[valid_tris_remove] -= 1
                counts[valid_tris_add] += 1
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                sol.append(to_add)
                sol_set.add(to_add)
                current_uncovered = new_uncovered
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)

        elif move_type < 0.95 and sol and len(sol) > 1:
            # 2-opt style: remove two blocks, add one block
            # Only try if solution is large enough
            if len(sol) >= 2:
                idx1, idx2 = rng.sample(range(len(sol)), 2)
                to_remove1 = sol[idx1]
                to_remove2 = sol[idx2]
                
                tris1 = block_triple_matrix[to_remove1]
                valid_tris1 = tris1[tris1 != -1]
                tris2 = block_triple_matrix[to_remove2]
                valid_tris2 = tris2[tris2 != -1]
                
                if len(valid_tris1) == 0 or len(valid_tris2) == 0:
                    continue
                
                # Simulate removal
                sim_counts = counts.copy()
                sim_counts[valid_tris1] -= 1
                sim_counts[valid_tris2] -= 1
                
                # Find triples that are now uncovered
                uncovered_mask = (sim_counts == 0)
                uncovered_tris = np.where(uncovered_mask)[0]
                
                if len(uncovered_tris) == 0:
                    # Pure removal of two blocks
                    new_uncovered = 0
                    new_energy = (len(sol) - 2) + 1 * 0
                    delta_e = new_energy - current_energy
                    if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                        counts[valid_tris1] -= 1
                        counts[valid_tris2] -= 1
                        sol.remove(to_remove1)
                        sol.remove(to_remove2)
                        sol_set.discard(to_remove1)
                        sol_set.discard(to_remove2)
                        current_uncovered = 0
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_sol = list(sol)
                else:
                    # Find a block that covers most uncovered_tris
                    candidate_blocks = set()
                    for t_idx in uncovered_tris[:10]:  # Limit to first 10 for speed
                        candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
                    candidate_blocks.difference_update(sol_set)
                    
                    if not candidate_blocks:
                        continue
                    
                    candidates_list = list(candidate_blocks)
                    if len(candidates_list) > 30:
                        candidates_list = rng.sample(candidates_list, 30)
                    
                    best_candidate = None
                    best_score = -1
                    for cand in candidates_list:
                        tris_cand = block_triple_matrix[cand]
                        valid_tris_cand = tris_cand[tris_cand != -1]
                        score = int(np.sum(np.isin(valid_tris_cand, uncovered_tris)))
                        if score > best_score:
                            best_score = score
                            best_candidate = cand
                    
                    if best_candidate is None or best_score == 0:
                        continue
                    
                    to_add = best_candidate
                    tris_add = block_triple_matrix[to_add]
                    valid_tris_add = tris_add[tris_add != -1]
                    
                    sim_counts[valid_tris_add] += 1
                    new_uncovered = int(np.sum(sim_counts == 0))
                    
                    new_w = 50 + 50 * new_uncovered if new_uncovered > 0 else 1
                    new_energy = (len(sol) - 1) + new_w * new_uncovered  # removed 2, added 1
                    delta_e = new_energy - current_energy
                    
                    if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                        counts[valid_tris1] -= 1
                        counts[valid_tris2] -= 1
                        counts[valid_tris_add] += 1
                        sol.remove(to_remove1)
                        sol.remove(to_remove2)
                        sol_set.discard(to_remove1)
                        sol_set.discard(to_remove2)
                        sol.append(to_add)
                        sol_set.add(to_add)
                        current_uncovered = new_uncovered
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_sol = list(sol)

        T *= cooling
        
        if time.time() > time_limit:
            break
            
    return best_sol

def _polish(sol, num_blocks, block_triple_matrix, block_triple_lens, 
            triple_to_blocks_arr, num_triples):
    """Deterministic polish pass: remove redundant blocks, then try targeted swaps."""
    if not sol:
        return sol
    
    sol_set = set(sol)
    
    # Pass 1: Remove redundant blocks
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_matrix[idx]
        valid_tris = tris[tris != -1]
        if len(valid_tris) > 0:
            counts[valid_tris] += 1
    
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            tris = block_triple_matrix[idx]
            valid_tris = tris[tris != -1]
            if len(valid_tris) == 0:
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
                continue
            if np.all(counts[valid_tris] > 1):
                counts[valid_tris] -= 1
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
    
    # Pass 2: Try to replace pairs of blocks with a single block
    for i in range(len(sol)):
        for j in range(i + 1, len(sol)):
            b1 = sol[i]
            b2 = sol[j]
            tris1 = block_triple_matrix[b1]
            valid_tris1 = tris1[tris1 != -1]
            tris2 = block_triple_matrix[b2]
            valid_tris2 = tris2[tris2 != -1]
            
            if len(valid_tris1) == 0 or len(valid_tris2) == 0:
                continue
            
            # Find triples that would be uncovered if we remove both
            sim_counts = counts.copy()
            sim_counts[valid_tris1] -= 1
            sim_counts[valid_tris2] -= 1
            uncovered_mask = (sim_counts == 0)
            uncovered_tris = np.where(uncovered_mask)[0]
            
            if len(uncovered_tris) == 0:
                # Both are redundant together
                counts[valid_tris1] -= 1
                counts[valid_tris2] -= 1
                sol.remove(b1)
                sol.remove(b2)
                sol_set.discard(b1)
                sol_set.discard(b2)
                i -= 1  # Adjust index after removal
                break
            elif len(uncovered_tris) <= 5:
                # Try to find a single block that covers all uncovered_tris
                candidate_blocks = set()
                for t_idx in uncovered_tris:
                    candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
                candidate_blocks.difference_update(sol_set)
                
                if candidate_blocks:
                    for cand in candidate_blocks:
                        tris_cand = block_triple_matrix[cand]
                        valid_tris_cand = tris_cand[tris_cand != -1]
                        if np.all(np.isin(uncovered_tris, valid_tris_cand)):
                            # This block covers all uncovered triples
                            counts[valid_tris1] -= 1
                            counts[valid_tris2] -= 1
                            counts[valid_tris_cand] += 1
                            sol.remove(b1)
                            sol.remove(b2)
                            sol_set.discard(b1)
                            sol_set.discard(b2)
                            sol.append(cand)
                            sol_set.add(cand)
                            i -= 1
                            break
                    else:
                        continue
                    break
    
    return sol
