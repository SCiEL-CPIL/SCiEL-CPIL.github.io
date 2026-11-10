import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free set modulo 205.
    Uses bitmask-accelerated local search with upper-bound pruning and multiple random starts.
    """
    m = 205
    deadline = time.time() + 570.0  # Safety margin for 600s limit
    
    # 1. Compute forbidden differences (non-zero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build conflict masks
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # 3. Precompute for each node, the list of non-conflicting nodes (excluding self)
    compatible = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            if not (conflict_mask[i] & (1 << j)):
                mask |= (1 << j)
        compatible[i] = mask
    
    # Precompute mutually compatible pairs for 2-swap
    mutually_compatible = [0] * m
    for i in range(m):
        mask = 0
        for j in range(i + 1, m):
            if compatible[i] & (1 << j):
                mask |= (1 << j)
        mutually_compatible[i] = mask

    def get_conflicts(selected_mask, candidate):
        """Returns True if candidate conflicts with any node in selected_mask."""
        return (selected_mask & conflict_mask[candidate]) != 0

    def greedy_upper_bound(available_mask, current_size):
        """
        Greedy coloring upper bound for the maximum independent set.
        """
        if available_mask == 0:
            return current_size
        
        colors = {}
        color_count = 0
        remaining = available_mask
        
        while remaining:
            color_count += 1
            color_class = 0
            temp_remaining = remaining
            while temp_remaining:
                node = (temp_remaining & -temp_remaining).bit_length() - 1
                color_class |= (1 << node)
                temp_remaining &= ~compatible[node] & ~(1 << node)
                temp_remaining &= remaining
            remaining &= ~color_class
        
        return current_size + color_count

    def greedy_random(seed):
        """Generate a random greedy solution."""
        random.seed(seed)
        selected_mask = 0
        candidates = list(range(m))
        random.shuffle(candidates)
        
        for c in candidates:
            if not get_conflicts(selected_mask, c):
                selected_mask |= (1 << c)
                
        return selected_mask

    def local_search(start_mask, max_time=5.0):
        """
        Improve a solution using 1-add, 2-swap, 3-swap moves with perturbation.
        """
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        no_improvement_count = 0
        start_time = time.time()
        shake_count = 0
        max_shakes = 30
        
        while no_improvement_count < 80 and shake_count < max_shakes:
            if time.time() - start_time > max_time or time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try 1-Add: Add a node that doesn't conflict
            for c in range(m):
                if not (current_mask & (1 << c)):
                    if not get_conflicts(current_mask, c):
                        current_mask |= (1 << c)
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        break
            
            if improved:
                no_improvement_count = 0
                continue
            
            current_indices = [i for i in range(m) if current_mask & (1 << i)]
            
            # 2. Try 2-Swap: Remove one node i, add two nodes j, k (net +1)
            found_2swap = False
            for i in current_indices:
                mask_without_i = current_mask & ~(1 << i)
                
                for j in range(m):
                    if current_mask & (1 << j):
                        continue
                    if mask_without_i & conflict_mask[j]:
                        continue
                    
                    # Use precomputed mutually compatible pairs
                    candidates_k = mutually_compatible[j] & ~current_mask & ~mask_without_i
                    candidates_k &= ~conflict_mask[j]  # j and k must be compatible
                    
                    if candidates_k:
                        k = (candidates_k & -candidates_k).bit_length() - 1
                        new_mask = mask_without_i | (1 << j) | (1 << k)
                        current_mask = new_mask
                        current_size += 1
                        improved = True
                        found_2swap = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        break
                if found_2swap:
                    break
            
            if found_2swap:
                no_improvement_count = 0
                continue
            
            # 3. Try 3-Swap: Remove two nodes i1, i2, add three nodes j, k, l (net +1)
            found_3swap = False
            if len(current_indices) >= 2:
                for idx1 in range(len(current_indices)):
                    i1 = current_indices[idx1]
                    for idx2 in range(idx1 + 1, len(current_indices)):
                        i2 = current_indices[idx2]
                        mask_without_ii = current_mask & ~(1 << i1) & ~(1 << i2)
                        
                        # Find candidates compatible with mask_without_ii
                        valid_candidates = []
                        for j in range(m):
                            if current_mask & (1 << j):
                                continue
                            if mask_without_ii & conflict_mask[j]:
                                continue
                            valid_candidates.append(j)
                        
                        if len(valid_candidates) >= 3:
                            # Try to find 3 mutually compatible candidates
                            found_triple = False
                            for a in range(len(valid_candidates)):
                                j = valid_candidates[a]
                                for b in range(a + 1, len(valid_candidates)):
                                    k = valid_candidates[b]
                                    if not (compatible[j] & (1 << k)):
                                        continue
                                    for c in range(b + 1, len(valid_candidates)):
                                        l = valid_candidates[c]
                                        if not (compatible[j] & (1 << l)) or not (compatible[k] & (1 << l)):
                                            continue
                                        
                                        new_mask = mask_without_ii | (1 << j) | (1 << k) | (1 << l)
                                        current_mask = new_mask
                                        current_size += 1
                                        improved = True
                                        found_3swap = True
                                        found_triple = True
                                        if current_size > best_size:
                                            best_size = current_size
                                            best_mask = current_mask
                                        break
                                    if found_triple:
                                        break
                                if found_triple:
                                    break
                            if found_3swap:
                                break
                    if found_3swap:
                        break
            
            if found_3swap:
                no_improvement_count = 0
                continue

            # 4. Try 1-Swap: Remove one node i, add one node j (size stays same)
            if current_indices:
                i = random.choice(current_indices)
                mask_without_i = current_mask & ~(1 << i)
                
                valid_js = []
                for j in range(m):
                    if current_mask & (1 << j):
                        continue
                    if not (mask_without_i & conflict_mask[j]):
                        valid_js.append(j)
                
                if valid_js:
                    j = random.choice(valid_js)
                    current_mask = mask_without_i | (1 << j)
                    no_improvement_count = 0
                    continue
            
            # If no moves found, we are stuck.
            no_improvement_count += 1
            
            # If stuck, perform a perturbation
            if no_improvement_count > 20:
                # Aggressive perturbation: remove 4-8 random nodes
                num_remove = random.randint(4, min(8, max(1, current_size - 2)))
                if current_size > num_remove:
                    indices_to_remove = random.sample(current_indices, num_remove)
                    for idx in indices_to_remove:
                        current_mask &= ~(1 << idx)
                    current_size = bin(current_mask).count('1')
                    no_improvement_count = 0
                    shake_count += 1
        
        return best_mask, best_size

    # Main Search Loop
    best_global_mask = 0
    best_global_size = 0
    
    num_starts = 300
    start_time = time.time()
    
    for i in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        remaining_time = deadline - time.time()
        if i < num_starts // 2:
            time_per_start = max(1.0, remaining_time / (num_starts - i))
        else:
            time_per_start = max(0.5, remaining_time / (num_starts - i))
        
        seed = i * 7919 + 104729
        init_mask = greedy_random(seed)
        
        improved_mask, improved_size = local_search(init_mask, max_time=min(time_per_start, 10.0))
        
        if improved_size > best_global_size:
            best_global_size = improved_size
            best_global_mask = improved_mask

    # Final polish: try to add more elements to the best solution
    if best_global_mask:
        changed = True
        while changed:
            changed = False
            for c in range(m):
                if not (best_global_mask & (1 << c)):
                    if not get_conflicts(best_global_mask, c):
                        best_global_mask |= (1 << c)
                        best_global_size += 1
                        changed = True

    # Convert best mask to array
    result = []
    for i in range(m):
        if best_global_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
