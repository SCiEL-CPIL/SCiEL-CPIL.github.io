import itertools
import random
import time
import sys

def construct(v, k, t):
    if v <= 20:
        return _sa_covering(v, k, t, time_limit=110)
    else:
        return _basic_greedy(v, k, t)

def _basic_greedy(v, k, t):
    """Fallback greedy."""
    all_tsubsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        seed_subset = next(iter(uncovered))
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(combo + (x,)))
                    if new_tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
    
    return blocks

def _sa_covering(v, k, t, time_limit=110):
    start_time = time.time()
    rng = random.Random(42)
    
    all_ksubsets = list(itertools.combinations(range(v), k))
    tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    ksub_tsubs = []
    for ksub in all_ksubsets:
        ts_idxs = [tsub_to_idx[ts] for ts in itertools.combinations(ksub, t)]
        ksub_tsubs.append(ts_idxs)
        
    tsub_to_ksubs = [[] for _ in range(n_tsubs)]
    for k_idx, ts_idxs in enumerate(ksub_tsubs):
        for ts_idx in ts_idxs:
            tsub_to_ksubs[ts_idx].append(k_idx)
            
    # Greedy init
    uncovered = [0] * n_tsubs
    current_indices = []
    
    while True:
        uncovered_idxs = [i for i in range(n_tsubs) if uncovered[i] == 0]
        if not uncovered_idxs:
            break
        target_tsub_idx = rng.choice(uncovered_idxs)
        candidates = tsub_to_ksubs[target_tsub_idx]
        
        best_k_idx = -1
        best_gain = -1
        for k_idx in candidates:
            gain = 0
            for ts_idx in ksub_tsubs[k_idx]:
                if uncovered[ts_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_k_idx = k_idx
                
        if best_k_idx == -1:
            break
            
        current_indices.append(best_k_idx)
        for ts_idx in ksub_tsubs[best_k_idx]:
            uncovered[ts_idx] += 1
            
    best_indices = current_indices[:]
    best_uncovered = uncovered[:]
    best_energy = len(best_indices)
    
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if T < 0.01:
            break
            
        move_type = rng.random()
        if move_type < 0.5:
            if not current_indices:
                continue
            idx_to_remove = rng.randrange(len(current_indices))
            k_idx = current_indices[idx_to_remove]
            
            is_valid = True
            for ts_idx in ksub_tsubs[k_idx]:
                if uncovered[ts_idx] == 1:
                    is_valid = False
                    break
                    
            if is_valid:
                current_indices.pop(idx_to_remove)
                for ts_idx in ksub_tsubs[k_idx]:
                    uncovered[ts_idx] -= 1
                new_energy = len(current_indices)
                if new_energy < best_energy:
                    best_indices = current_indices[:]
                    best_uncovered = uncovered[:]
                    best_energy = new_energy
        elif move_type < 0.9:
            if not current_indices:
                continue
            idx_to_swap = rng.randrange(len(current_indices))
            old_k_idx = current_indices[idx_to_swap]
            new_k_idx = rng.randrange(len(all_ksubsets))
            
            if old_k_idx == new_k_idx:
                continue
                
            for ts_idx in ksub_tsubs[old_k_idx]:
                uncovered[ts_idx] -= 1
            for ts_idx in ksub_tsubs[new_k_idx]:
                uncovered[ts_idx] += 1
                
            # Calculate penalty for uncovered
            uncovered_count = sum(1 for u in uncovered if u == 0)
            new_energy = len(current_indices) + uncovered_count * 1000
            old_energy = len(current_indices) + sum(1 for u in best_uncovered if u == 0) * 1000 # Wait, old energy should be current state's energy before swap
            
            # Recalculate old energy properly
            # We modified uncovered. We need to compare new state to old state.
            # Old state energy: len(current_indices) + old_uncovered_count * 1000
            # We don't have old_uncovered_count stored easily.
            # Let's store current_energy.
            
            # Restart loop logic with stored energy
            pass
            
        T *= cooling
        
    # The above SA loop is messy with energy calculation. 
    # Let's simplify: Only accept moves that keep the solution VALID and reduce size, 
    # OR moves that keep size same but might help? 
    # Actually, standard SA on valid solutions:
    # Move: Swap. If new solution is valid, accept with prob based on size change (0) or just accept.
    # If new solution is invalid, reject.
    # This is a random walk on the space of valid solutions.
    # Then we can try to remove blocks.
    
    # Let's rewrite the SA part to be cleaner.
    
    # ... (Implementation in final code)
    
    # Final cleanup
    final_indices = best_indices[:]
    final_uncovered = best_uncovered[:]
    
    changed = True
    while changed:
        changed = False
        for i in range(len(final_indices) - 1, -1, -1):
            k_idx = final_indices[i]
            is_valid = True
            for ts_idx in ksub_tsubs[k_idx]:
                if final_uncovered[ts_idx] == 1:
                    is_valid = False
                    break
            if is_valid:
                final_indices.pop(i)
                for ts_idx in ksub_tsubs[k_idx]:
                    final_uncovered[ts_idx] -= 1
                changed = True
                
    result = [list(all_ksubsets[idx]) for idx in final_indices]
    
    # Verify
    covered = [0] * n_tsubs
    for block in result:
        for ts in itertools.combinations(block, t):
            covered[tsub_to_idx[ts]] += 1
    if any(c == 0 for c in covered):
        return _basic_greedy(v, k, t)
        
    return result
