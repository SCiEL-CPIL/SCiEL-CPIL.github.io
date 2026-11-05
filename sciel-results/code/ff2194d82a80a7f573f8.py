import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    t_subsets_per_block = math.comb(k, t)

    # Precompute block -> triple indices
    # Each block covers t_subsets_per_block triples
    block_triple_indices = []
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices.append(np.array(tris, dtype=np.int32))

    # Convert triple_to_blocks to numpy arrays for faster access
    triple_to_blocks_arr = [np.array(b, dtype=np.int32) for b in triple_to_blocks]
    
    # Frequency of each triple across all blocks (for weighted greedy)
    # Count how many blocks cover each triple
    freq = np.zeros(num_triples, dtype=np.int32)
    for tris in block_triple_indices:
        freq[tris] += 1
    inv_freq = 1.0 / (freq.astype(float) + 1e-9)

    best_sol = None
    best_energy = float('inf')

    seed_counter = 0
    restarts = 0

    while time.time() < deadline - 15:
        seed_counter += 1
        restarts += 1
        rng = random.Random(seed_counter)

        # Cycle through strategies: 0=greedy, 1=weighted, 2=randomized
        strategy = seed_counter % 3

        if strategy == 0:
            sol = _fast_greedy(rng, num_blocks, block_triple_indices, num_triples, deadline - 15, mode='greedy')
        elif strategy == 1:
            sol = _fast_greedy(rng, num_blocks, block_triple_indices, num_triples, deadline - 15, mode='weighted', inv_freq=inv_freq)
        else:
            sol = _fast_greedy(rng, num_blocks, block_triple_indices, num_triples, deadline - 15, mode='randomized')

        # Remove Redundant Blocks
        sol = _remove_redundant(sol, block_triple_indices, num_triples)

        # Local Search (SA)
        time_left_for_sa = deadline - 15 - time.time()
        if time_left_for_sa < 5:
            break

        sol = _local_search(sol, num_blocks, block_triple_indices, triple_to_blocks_arr, num_triples, time_left_for_sa, rng)

        # Final polish: deterministic removal + targeted swap
        sol = _polish(sol, num_blocks, block_triple_indices, triple_to_blocks_arr, num_triples)

        # Evaluate
        if sol:
            covered = np.zeros(num_triples, dtype=bool)
            for b_idx in sol:
                tris = block_triple_indices[b_idx]
                if len(tris) > 0:
                    covered[tris] = True

            uncovered_count = np.sum(~covered)

            if uncovered_count == 0:
                energy = len(sol)
                if energy < best_energy:
                    best_energy = energy
                    best_sol = list(sol)
                    # Optional: break if we hit a very good solution? 
                    # Best known is 76. If we get 76, it's good.
                    # But maybe we can find 75? Unlikely but possible.
                    # Let's keep searching if time permits.

    if best_sol is None:
        # Fallback: return a valid greedy solution if no valid one found in loop (should not happen)
        sol = _fast_greedy(random.Random(0), num_blocks, block_triple_indices, num_triples, deadline - 1, mode='greedy')
        sol = _remove_redundant(sol, block_triple_indices, num_triples)
        return [list(all_blocks[i]) for i in sol]

    # Convert indices back to point lists
    return [list(all_blocks[i]) for i in best_sol]

def _fast_greedy(rng, num_blocks, block_triple_indices, num_triples, deadline, mode='greedy', inv_freq=None):
    """Fast greedy with randomized tie-breaking."""
    uncovered = np.ones(num_triples, dtype=bool)
    selected = []
    selected_set = set()

    # Precompute gains array
    # gains[i] = number of uncovered triples covered by block i
    
    # We will update gains incrementally or recompute? 
    # Recalculating from scratch each step is O(N_blocks * t_subsets). 
    # With 346k blocks and 21 triples per block, it's ~7M ops per step.
    # Greedy might take ~50-100 steps. 700M ops total. Might be slow in Python loop.
    # Let's use numpy vectorization for gain calculation.
    
    while np.any(uncovered) and time.time() < deadline:
        # Calculate gains for all blocks
        # For each block, count how many of its triples are in 'uncovered'
        # block_triple_indices is a list of arrays.
        # We can do this in a loop but it's slow. 
        # Alternative: maintain a matrix? Too big (346k x 2024).
        
        # Let's optimize the gain calculation.
        # gains = np.array([np.sum(uncovered[tris]) for tris in block_triple_indices])
        # This is a python loop over 346k items. ~0.1-0.2s per step? 
        # 100 steps = 10-20s. Too slow.
        
        # We need a faster way.
        # If we maintain 'uncovered' as a vector, and we have block_triple_indices,
        # we can't easily vectorize across the list of variable length arrays (though they are fixed length here).
        
        # Actually, block_triple_indices are all same length (21).
        # So we can make it a 2D numpy array!
        # But we passed a list. Let's convert once outside? 
        # No, _fast_greedy is called multiple times.
        # Let's assume block_triple_indices is a list of numpy arrays.
        
        # To speed up, we can precompute a 2D array in construct and pass it.
        # But for now, let's try to make the loop faster or use a different approach.
        
        # Let's use the fact that we can update 'uncovered' and calculate gains.
        # If we just do the loop, it might be okay for a few restarts if we limit steps.
        # But 346k iterations in Python is ~10ms. 100 steps = 1s. That's acceptable.
        
        gains = np.zeros(num_blocks, dtype=np.int32)
        for i, tris in enumerate(block_triple_indices):
            if i in selected_set:
                gains[i] = -1
            else:
                gains[i] = np.sum(uncovered[tris])

        best_gain = np.max(gains)

        if best_gain <= 0:
            break

        # Randomized selection among top candidates
        candidates = np.where(gains == best_gain)[0]
        if len(candidates) > 1:
            if mode == 'randomized':
                chosen = int(rng.choice(candidates))
            else:
                # For greedy/weighted, pick the first one (deterministic) or random if weighted?
                # Weighted greedy should pick max weight, not just count.
                # My current 'gains' is just count. 
                # For weighted, I need to sum inv_freq[uncovered[tris]].
                
                if mode == 'weighted':
                     # Recalculate gains with weights
                     w_gains = np.zeros(num_blocks, dtype=np.float32)
                     for i, tris in enumerate(block_triple_indices):
                         if i in selected_set:
                             w_gains[i] = -1
                         else:
                             w_gains[i] = np.sum(inv_freq[uncovered[tris]])
                     best_w_gain = np.max(w_gains)
                     if best_w_gain <= 0: break
                     w_candidates = np.where(w_gains == best_w_gain)[0]
                     chosen = int(w_candidates[0])
                else:
                     chosen = int(candidates[0])
        else:
            chosen = int(candidates[0])

        selected.append(chosen)
        selected_set.add(chosen)

        # Update uncovered
        tris = block_triple_indices[chosen]
        uncovered[tris] = False

    return selected

def _remove_redundant(sol, block_triple_indices, num_triples):
    if not sol:
        return sol

    # Count coverage
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_indices[idx]
        if len(tris) > 0:
            counts[tris] += 1

    sol_set = set(sol)
    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            tris = block_triple_indices[idx]
            if len(tris) == 0:
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
                continue
            if np.any(counts[tris] == 1):
                continue
            # Redundant
            counts[tris] -= 1
            sol.remove(idx)
            sol_set.discard(idx)
            changed = True

    return sol

def _local_search(initial_sol, num_blocks, block_triple_indices, triple_to_blocks_arr, num_triples, time_limit, rng):
    sol = list(initial_sol)
    sol_set = set(sol)

    # Coverage counts
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_indices[idx]
        if len(tris) > 0:
            counts[tris] += 1

    current_uncovered = int(np.sum(counts == 0))
    
    def get_energy():
        # Penalty for uncovered triples is huge
        return len(sol) + 1000 * current_uncovered

    current_energy = get_energy()
    best_sol = list(sol)
    best_energy = current_energy

    T = 5.0
    cooling = 0.995
    min_T = 0.1

    while time.time() < time_limit and T > min_T:
        move_type = rng.random()

        if move_type < 0.4 and sol:
            # Removal
            to_remove = rng.choice(sol)
            tris = block_triple_indices[to_remove]
            if len(tris) == 0:
                continue

            min_count_in_block = int(np.min(counts[tris]))

            if min_count_in_block > 1:
                # Redundant
                counts[tris] -= 1
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                current_uncovered = int(np.sum(counts == 0))
                current_energy = get_energy()
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)
            else:
                delta_uncovered = int(np.sum(counts[tris] == 1))
                new_uncovered = current_uncovered + delta_uncovered
                new_energy = (len(sol) - 1) + 1000 * new_uncovered
                delta_e = new_energy - current_energy

                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    counts[tris] -= 1
                    sol.remove(to_remove)
                    sol_set.discard(to_remove)
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)

        elif move_type < 0.8 and sol:
            # Swap
            to_remove = rng.choice(sol)
            tris_remove = block_triple_indices[to_remove]
            if len(tris_remove) == 0:
                continue

            # Find triples that will become uncovered
            uncovered_mask = (counts[tris_remove] == 1)
            uncovered_tris = tris_remove[uncovered_mask]

            if len(uncovered_tris) == 0:
                # Pure removal, handled in removal logic, but let's allow swap to same size
                # If no uncovered, it's a removal.
                continue

            # Find candidate blocks that cover at least one uncovered triple
            candidate_blocks = set()
            for t_idx in uncovered_tris:
                candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
            candidate_blocks.difference_update(sol_set)

            if not candidate_blocks:
                continue

            candidates_list = list(candidate_blocks)
            if len(candidates_list) > 50:
                candidates_list = rng.sample(candidates_list, 50)

            best_candidate = None
            best_score = -1
            for cand in candidates_list:
                tris_add = block_triple_indices[cand]
                score = int(np.sum(np.isin(tris_add, uncovered_tris)))
                if score > best_score:
                    best_score = score
                    best_candidate = cand

            if best_candidate is None or best_score == 0:
                continue

            to_add = best_candidate
            tris_add = block_triple_indices[to_add]

            new_counts = counts.copy()
            new_counts[tris_remove] -= 1
            new_counts[tris_add] += 1
            new_uncovered = int(np.sum(new_counts == 0))

            new_energy = len(sol) + 1000 * new_uncovered # len unchanged
            delta_e = new_energy - current_energy

            if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                counts[tris_remove] -= 1
                counts[tris_add] += 1
                sol.remove(to_remove)
                sol_set.discard(to_remove)
                sol.append(to_add)
                sol_set.add(to_add)
                current_uncovered = new_uncovered
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_sol = list(sol)

        elif move_type < 0.95 and len(sol) >= 2:
            # 2-Swap: Remove 2, Add 1
            idx1, idx2 = rng.sample(range(len(sol)), 2)
            to_remove1 = sol[idx1]
            to_remove2 = sol[idx2]

            tris1 = block_triple_indices[to_remove1]
            tris2 = block_triple_indices[to_remove2]

            if len(tris1) == 0 or len(tris2) == 0:
                continue

            sim_counts = counts.copy()
            sim_counts[tris1] -= 1
            sim_counts[tris2] -= 1

            uncovered_mask = (sim_counts == 0)
            uncovered_tris = np.where(uncovered_mask)[0]

            if len(uncovered_tris) == 0:
                # Pure removal of 2
                new_uncovered = 0
                new_energy = (len(sol) - 2) + 1000 * 0
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    counts[tris1] -= 1
                    counts[tris2] -= 1
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
                for t_idx in uncovered_tris[:10]:
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
                    tris_cand = block_triple_indices[cand]
                    score = int(np.sum(np.isin(tris_cand, uncovered_tris)))
                    if score > best_score:
                        best_score = score
                        best_candidate = cand

                if best_candidate is None or best_score == 0:
                    continue

                to_add = best_candidate
                tris_add = block_triple_indices[to_add]

                sim_counts[tris_add] += 1
                new_uncovered = int(np.sum(sim_counts == 0))

                new_energy = (len(sol) - 1) + 1000 * new_uncovered
                delta_e = new_energy - current_energy

                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    counts[tris1] -= 1
                    counts[tris2] -= 1
                    counts[tris_add] += 1
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

    return best_sol

def _polish(sol, num_blocks, block_triple_indices, triple_to_blocks_arr, num_triples):
    if not sol:
        return sol

    sol_set = set(sol)

    # Pass 1: Remove redundant blocks
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        tris = block_triple_indices[idx]
        if len(tris) > 0:
            counts[tris] += 1

    changed = True
    while changed:
        changed = False
        for idx in list(sol):
            tris = block_triple_indices[idx]
            if len(tris) == 0:
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True
                continue
            if np.all(counts[tris] > 1):
                counts[tris] -= 1
                sol.remove(idx)
                sol_set.discard(idx)
                changed = True

    # Pass 2: Try to replace pairs of blocks with a single block
    # This is expensive O(N^2). Only do if sol is small enough.
    if len(sol) > 100:
        return sol

    for i in range(len(sol)):
        for j in range(i + 1, len(sol)):
            b1 = sol[i]
            b2 = sol[j]
            tris1 = block_triple_indices[b1]
            tris2 = block_triple_indices[b2]

            if len(tris1) == 0 or len(tris2) == 0:
                continue

            sim_counts = counts.copy()
            sim_counts[tris1] -= 1
            sim_counts[tris2] -= 1
            uncovered_mask = (sim_counts == 0)
            uncovered_tris = np.where(uncovered_mask)[0]

            if len(uncovered_tris) == 0:
                # Both redundant
                counts[tris1] -= 1
                counts[tris2] -= 1
                sol.remove(b1)
                sol.remove(b2)
                sol_set.discard(b1)
                sol_set.discard(b2)
                i -= 1
                break
            elif len(uncovered_tris) <= 5:
                # Try to find a single block that covers all uncovered_tris
                candidate_blocks = set()
                for t_idx in uncovered_tris:
                    candidate_blocks.update(triple_to_blocks_arr[t_idx].tolist())
                candidate_blocks.difference_update(sol_set)

                if candidate_blocks:
                    for cand in candidate_blocks:
                        tris_cand = block_triple_indices[cand]
                        if np.all(np.isin(uncovered_tris, tris_cand)):
                            counts[tris1] -= 1
                            counts[tris2] -= 1
                            counts[tris_cand] += 1
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
