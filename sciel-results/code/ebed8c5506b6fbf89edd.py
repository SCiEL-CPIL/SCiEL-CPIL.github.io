import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)
    
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] contains indices of t-subsets covered by blocks[i]
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    # triple_freq: how many blocks cover each triple
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    
    # triple_to_blocks: list of block indices for each triple
    # This is useful for the local search
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[i, idx] = t_idx
            triple_freq[t_idx] += 1
            triple_to_blocks[t_idx].append(i)
            idx += 1
            
    # Weights: 1 / frequency
    weights = 1.0 / np.maximum(triple_freq, 1)
    
    def is_valid_cover(block_indices):
        if not block_indices:
            return n_triples == 0
        covered = np.zeros(n_triples, dtype=bool)
        for b_idx in block_indices:
            covered[block_triples[b_idx]] = True
        return np.all(covered)
        
    def prune_redundant(block_indices):
        """Remove blocks that are completely redundant."""
        if not block_indices:
            return []
        
        current_selected = list(block_indices)
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in current_selected:
            cover_count[block_triples[b_idx]] += 1
            
        changed = True
        while changed:
            changed = False
            for i in range(len(current_selected)):
                b_idx = current_selected[i]
                # If all triples in this block are covered by at least 2 blocks, it's redundant
                if np.min(cover_count[block_triples[b_idx]]) > 1:
                    cover_count[block_triples[b_idx]] -= 1
                    current_selected.pop(i)
                    changed = True
                    break # Restart loop
                    
        return current_selected

    def run_greedy(use_weighted=True, random_tie=False, seed=None):
        if seed is not None:
            random.seed(seed)
            
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if use_weighted:
                # Calculate gains
                # To avoid huge memory spikes, we can do this in chunks or just rely on NumPy
                # The temp array is ~50MB, which is fine.
                gains = np.sum(uncovered[block_triples].astype(np.float64) * weights[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)
                
            if gains.max() == 0:
                break
                
            if random_tie:
                max_gain = gains.max()
                candidates = np.where(gains == max_gain)[0]
                best_idx = random.choice(candidates.tolist())
            else:
                best_idx = np.argmax(gains)
                
            selected_blocks.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected_blocks

    def local_search(block_indices, max_time=30.0):
        """
        Improve the solution using 2-for-1 exchanges and pruning.
        """
        current = list(block_indices)
        if not current:
            return current
            
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in current:
            cover_count[block_triples[b_idx]] += 1
            
        current_set = set(current)
        
        start_ls = time.time()
        improved = True
        while improved and time.time() < start_ls + max_time:
            improved = False
            
            # 1. Prune
            pruned = prune_redundant(current)
            if len(pruned) < len(current):
                # Rebuild cover_count and set
                current = pruned
                current_set = set(current)
                cover_count = np.zeros(n_triples, dtype=np.int32)
                for b_idx in current:
                    cover_count[block_triples[b_idx]] += 1
                improved = True
                continue # Restart loop as structure changed significantly

            # 2. 2-for-1 Exchange
            # For each pair (i, j) in current:
            #   Find U: triples that become uncovered if i, j removed.
            #   If U is not empty, check if any block C not in current covers U.
            n_cur = len(current)
            found_swap = False
            
            for i_idx in range(n_cur):
                if time.time() > deadline - 5.0:
                    return current
                b1 = current[i_idx]
                for j_idx in range(i_idx + 1, n_cur):
                    b2 = current[j_idx]
                    
                    # Identify U
                    # Triples covered by b1 or b2
                    # We need to check cover_count.
                    # If we remove b1 and b2, a triple t is uncovered if:
                    # cover_count[t] == 1 and (t in b1 or t in b2)
                    # OR cover_count[t] == 2 and (t in b1 and t in b2)
                    
                    # Efficient way:
                    # Iterate through triples in b1 and b2.
                    # Since c_k_t is small (35), this is fast.
                    
                    u_triples = []
                    
                    # Check b1's triples
                    for t_idx in block_triples[b1]:
                        if cover_count[t_idx] == 1:
                            u_triples.append(t_idx)
                        elif cover_count[t_idx] == 2:
                            # Check if b2 also covers it
                            # We can check if t_idx is in block_triples[b2]
                            # Since block_triples[b2] is small, linear search or set?
                            # Let's use a set for b2's triples for this check
                            pass # We'll handle this in a combined loop or set intersection later
                
                    # Better:
                    # Set of triples in b1
                    set_b1 = set(block_triples[b1].tolist())
                    set_b2 = set(block_triples[b2].tolist())
                    
                    u_triples = []
                    # Triples in b1
                    for t_idx in set_b1:
                        if cover_count[t_idx] == 1:
                            u_triples.append(t_idx)
                        elif cover_count[t_idx] == 2 and t_idx in set_b2:
                            u_triples.append(t_idx)
                    
                    # Triples in b2 (but not b1, to avoid duplicates)
                    for t_idx in set_b2:
                        if t_idx not in set_b1:
                            if cover_count[t_idx] == 1:
                                u_triples.append(t_idx)
                            elif cover_count[t_idx] == 2 and t_idx in set_b1:
                                # Already handled? No, if t in b2 and t in b1, it's in set_b1.
                                # So we only need to check t in b2 \ b1.
                                pass
                            # If t in b2 \ b1 and cover_count == 2, it must be covered by b2 and some other block k.
                            # Removing b1, b2 leaves k. So it's NOT uncovered.
                            # So only count==1 matters for b2 \ b1.
                    
                    if not u_triples:
                        # We can remove both!
                        # This should have been caught by prune_redundant?
                        # If u_triples is empty, it means all triples in b1 and b2 are covered by other blocks.
                        # Then b1 is redundant. Prune should have removed it.
                        # So this case should not occur if prune_redundant is robust.
                        # But if it does, we remove both.
                        current.remove(b1)
                        current.remove(b2)
                        current_set.discard(b1)
                        current_set.discard(b2)
                        cover_count[block_triples[b1]] -= 1
                        cover_count[block_triples[b2]] -= 1
                        improved = True
                        found_swap = True
                        break
                    
                    # Find C not in current_set such that u_triples subset C
                    # Intersection of triple_to_blocks for t in u_triples
                    if len(u_triples) == 1:
                        candidates = triple_to_blocks[u_triples[0]]
                    else:
                        # Start with the first
                        candidates = set(triple_to_blocks[u_triples[0]])
                        for t in u_triples[1:]:
                            candidates.intersection_update(triple_to_blocks[t])
                            if not candidates:
                                break
                    
                    if candidates:
                        # Filter out current blocks
                        valid_candidates = [c for c in candidates if c not in current_set]
                        if valid_candidates:
                            # Pick one (e.g., the one with max gain? or just first)
                            c_new = valid_candidates[0]
                            
                            # Perform swap: remove b1, b2; add c_new
                            current.remove(b1)
                            current.remove(b2)
                            current.append(c_new)
                            current_set.discard(b1)
                            current_set.discard(b2)
                            current_set.add(c_new)
                            
                            # Update cover_count
                            cover_count[block_triples[b1]] -= 1
                            cover_count[block_triples[b2]] -= 1
                            cover_count[block_triples[c_new]] += 1
                            
                            improved = True
                            found_swap = True
                            break
                            
                if found_swap:
                    break
                    
        return current

    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Standard Weighted Greedy
    for strategy in [
        {'use_weighted': True, 'random_tie': False, 'seed': None},
        {'use_weighted': False, 'random_tie': False, 'seed': None},
    ]:
        if time.time() > deadline - 20.0:
            continue
            
        selected = run_greedy(**strategy)
        pruned = prune_redundant(selected)
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(blocks[i]) for i in pruned]
            
        if best_size <= 69:
            break

    # Strategy 2: Randomized Restart Greedy + Local Search
    restart_count = 0
    while time.time() < deadline - 15.0:
        restart_count += 1
        seed = restart_count * 104729 + 1
        use_weighted = (restart_count % 2 == 0)
        
        selected = run_greedy(use_weighted=use_weighted, random_tie=True, seed=seed)
        pruned = prune_redundant(selected)
        
        # Apply Local Search to the pruned solution
        # Limit time for local search per restart
        ls_time_limit = min(10.0, max(1.0, deadline - time.time() - 15.0))
        improved = local_search(pruned, max_time=ls_time_limit)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = [list(blocks[i]) for i in improved]
            
        if best_size <= 69:
            break

    # Final Polish on the best solution found
    if best_solution is not None and time.time() < deadline - 2.0:
        block_to_idx = {tuple(b): i for i, b in enumerate(blocks)}
        current_indices = []
        for b in best_solution:
            current_indices.append(block_to_idx[tuple(b)])
            
        # Run local search again with remaining time
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 1.0:
            improved = local_search(current_indices, max_time=remaining_time)
            if len(improved) < best_size:
                best_size = len(improved)
                best_solution = [list(blocks[i]) for i in improved]

    if best_solution is None:
        # Fallback
        selected = run_greedy(use_weighted=True, random_tie=False)
        best_solution = [list(blocks[i]) for i in selected]
        
    return best_solution
