import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)
    
    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)
    
    # Precompute as numpy arrays for faster access
    k_sub_t_idxs_np = [np.array(idx, dtype=np.int32) for idx in k_sub_t_idxs]
    t_sub_to_k_idxs_np = [np.array(idx, dtype=np.int32) for idx in t_sub_to_k_idxs]

    def greedy_cover(seed):
        """
        Efficient greedy set cover using NumPy for gain updates.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # Initial gains: number of t-subsets covered by each block
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        
        selected = []
        
        for _ in range(n_k): # Safety bound
            if not uncovered.any():
                break
            
            # Find best block
            if seed > 0:
                # Add small noise to break ties
                scores = gains.astype(np.float64) + rng.random() * 0.01
            else:
                scores = gains.astype(np.float64)
            
            best_idx = int(np.argmax(scores))
            
            if scores[best_idx] < 0.5: # Gain is integer, so < 0.5 means 0
                break
                
            selected.append(best_idx)
            
            # Update gains efficiently
            t_idxs = k_sub_t_idxs_np[best_idx]
            # Find which of these t-subsets are still uncovered
            mask = uncovered[t_idxs]
            newly_covered = t_idxs[mask]
            
            if len(newly_covered) > 0:
                uncovered[newly_covered] = False
                # Decrease gain for all blocks covering these newly-covered t-subsets
                # Use vectorized update where possible
                for t_idx in newly_covered:
                    gains[t_sub_to_k_idxs_np[t_idx]] -= 1
                        
        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            cover_count[k_sub_t_idxs_np[b_idx]] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                t_idxs = k_sub_t_idxs_np[b_idx]
                if not np.any(cover_count[t_idxs] == 1):
                    changed = True
                    cover_count[t_idxs] -= 1
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks

        return blocks_idx

    def local_search(initial_blocks, max_time):
        """
        Local search with remove and swap moves, plus aggressive reduction.
        """
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[k_sub_t_idxs_np[b_idx]] += 1
        
        current_energy = float(len(current_blocks))
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        # Parameters
        T = 30.0
        cooling_rate = 0.995
        min_T = 0.05
        penalty = 500.0
        
        rng = random.Random(123)
        
        # Track blocks by index for O(1) lookup
        block_set = set(current_blocks)
        
        while time.time() < max_time and T > min_T:
            move_type = rng.random()
            
            # --- Move 1: Try to Remove ---
            if move_type < 0.3 and len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                t_idxs = k_sub_t_idxs_np[b_idx]
                
                # Check if removal is valid
                min_cover = np.min(cover_count[t_idxs])
                if min_cover > 1:
                    # Valid removal
                    cover_count[t_idxs] -= 1
                    current_blocks.pop(i)
                    block_set.discard(b_idx)
                    current_energy -= 1.0
                else:
                    # Invalid: count how many become uncovered
                    new_uncovered = int(np.sum(cover_count[t_idxs] == 1))
                    if new_uncovered == 0:
                        # Actually valid
                        cover_count[t_idxs] -= 1
                        current_blocks.pop(i)
                        block_set.discard(b_idx)
                        current_energy -= 1.0
                    else:
                        delta_E = 1.0 + penalty * new_uncovered
                        if rng.random() < np.exp(-delta_E / T):
                            cover_count[t_idxs] -= 1
                            current_blocks.pop(i)
                            block_set.discard(b_idx)
                            current_energy += delta_E

            # --- Move 2: Try to Swap ---
            elif move_type < 0.7 and len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                old_t_idxs = k_sub_t_idxs_np[old_b]
                
                # Find unique subsets covered by old_b
                unique_mask = cover_count[old_t_idxs] == 1
                unique_ts = old_t_idxs[unique_mask]
                
                if len(unique_ts) == 0:
                    # No unique subsets. Swap with random other block.
                    new_b = rng.randint(0, n_k - 1)
                    if new_b in block_set:
                        continue
                    
                    # Perform swap
                    cover_count[old_t_idxs] -= 1
                    new_t_idxs = k_sub_t_idxs_np[new_b]
                    cover_count[new_t_idxs] += 1
                    
                    current_blocks[i] = new_b
                    block_set.discard(old_b)
                    block_set.add(new_b)
                    
                else:
                    # Must cover all unique subsets
                    # Find candidates that cover all unique_ts
                    if len(unique_ts) == 1:
                        candidates = t_sub_to_k_idxs_np[unique_ts[0]]
                    else:
                        # Intersect candidate sets
                        candidate_set = set(t_sub_to_k_idxs_np[unique_ts[0]])
                        for t_idx in unique_ts[1:]:
                            candidate_set &= set(t_sub_to_k_idxs_np[t_idx])
                        if not candidate_set:
                            continue
                        candidates = np.array(list(candidate_set), dtype=np.int32)
                    
                    # Remove current blocks
                    candidates = candidates[~np.isin(candidates, list(block_set))]
                    if len(candidates) == 0:
                        continue
                    
                    new_b = int(rng.choice(candidates))
                    
                    # Perform swap
                    cover_count[old_t_idxs] -= 1
                    new_t_idxs = k_sub_t_idxs_np[new_b]
                    cover_count[new_t_idxs] += 1
                    
                    current_blocks[i] = new_b
                    block_set.discard(old_b)
                    block_set.add(new_b)

            # --- Move 3: Two-for-one swap (remove one, add two that cover its unique parts) ---
            elif move_type < 0.85 and len(current_blocks) > 0:
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                old_t_idxs = k_sub_t_idxs_np[old_b]
                
                unique_mask = cover_count[old_t_idxs] == 1
                unique_ts = old_t_idxs[unique_mask]
                
                if len(unique_ts) == 0:
                    continue
                
                # Try to find two blocks that together cover all unique_ts
                # and have good overlap with each other to minimize total added subsets
                if len(unique_ts) <= 2:
                    # For small unique sets, try all pairs
                    if len(unique_ts) == 1:
                        cand1 = t_sub_to_k_idxs_np[unique_ts[0]]
                        cand1 = cand1[~np.isin(cand1, list(block_set))]
                        if len(cand1) < 2:
                            continue
                        # Pick two random candidates
                        idxs = rng.sample(range(len(cand1)), 2)
                        b1 = int(cand1[idxs[0]])
                        b2 = int(cand1[idxs[1]])
                    else:
                        # Need blocks that cover different unique_ts
                        cand1 = t_sub_to_k_idxs_np[unique_ts[0]]
                        cand2 = t_sub_to_k_idxs_np[unique_ts[1]]
                        cand1 = cand1[~np.isin(cand1, list(block_set))]
                        cand2 = cand2[~np.isin(cand2, list(block_set))]
                        if len(cand1) == 0 or len(cand2) == 0:
                            continue
                        b1 = int(rng.choice(cand1))
                        b2 = int(rng.choice(cand2))
                        if b1 == b2:
                            continue
                else:
                    continue
                
                # Check if b1 and b2 together cover all unique_ts
                t1 = k_sub_t_idxs_np[b1]
                t2 = k_sub_t_idxs_np[b2]
                covered = set(t1.tolist()) | set(t2.tolist())
                if not all(t in covered for t in unique_ts):
                    continue
                
                # Calculate net change in covered subsets
                # Remove old_b, add b1 and b2
                # Net gain = |unique_ts covered by b1∪b2 that weren't covered by old_b| - |other subsets newly covered|
                # Simplified: just accept if it doesn't create new uncovered
                # Since b1∪b2 covers all unique_ts, no new uncovered from old_b's removal
                # Check if b1 or b2 create any issues? No, adding blocks only helps.
                
                # Perform move
                cover_count[old_t_idxs] -= 1
                cover_count[t1] += 1
                cover_count[t2] += 1
                
                current_blocks.pop(i)
                current_blocks.extend([b1, b2])
                block_set.discard(old_b)
                block_set.add(b1)
                block_set.add(b2)
                
                # Energy: +1 block (removed 1, added 2)
                current_energy += 1.0

            # --- Aggressive reduction pass ---
            # Every few iterations, do a thorough reduction
            if int((T - min_T) * 100) % 50 == 0 or T < 1.0:
                changed = True
                while changed:
                    changed = False
                    for j in range(len(current_blocks)):
                        b_idx = current_blocks[j]
                        t_idxs = k_sub_t_idxs_np[b_idx]
                        if not np.any(cover_count[t_idxs] == 1):
                            cover_count[t_idxs] -= 1
                            current_blocks.pop(j)
                            block_set.discard(b_idx)
                            current_energy -= 1.0
                            changed = True
                            break

            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)

            # Cool down
            T *= cooling_rate

        # Final reduction
        changed = True
        while changed:
            changed = False
            for j in range(len(current_blocks)):
                b_idx = current_blocks[j]
                t_idxs = k_sub_t_idxs_np[b_idx]
                if not np.any(cover_count[t_idxs] == 1):
                    cover_count[t_idxs] -= 1
                    current_blocks.pop(j)
                    block_set.discard(b_idx)
                    current_energy -= 1.0
                    changed = True
                    break

        return current_blocks if len(current_blocks) < len(best_blocks) else best_blocks

    # Run initial greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    
    best_blocks = blocks_idx
    best_len = len(blocks_idx)
    
    # Try randomized greedy if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
        for seed in seeds:
            if time.time() >= deadline - 10.0:
                break
            b_idx = greedy_cover(seed)
            b_idx = reduce_blocks(b_idx)
            if b_idx and len(b_idx) < best_len:
                best_blocks = b_idx
                best_len = len(b_idx)

    # Run local search on the best solution found
    if best_blocks and time.time() < deadline - 5.0:
        ls_result = local_search(best_blocks, deadline - 5.0)
        if len(ls_result) < best_len:
            best_blocks = ls_result
            best_len = len(best_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
