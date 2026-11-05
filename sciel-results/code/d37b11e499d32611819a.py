import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 55  # seconds, keep a margin
    
    # 1. Generate the core set: numbers with base-3 digits 0 or 1
    # This is a known 4-AP-free set (Cantor set)
    core = set()
    for mask in range(1, 1 << 6):  # 6 digits: 3^0 to 3^5
        val = 0
        for i in range(6):
            if (mask >> i) & 1:
                val += (3 ** i)
        if 1 <= val <= n:
            core.add(val)
    
    # 2. Precompute conflict masks for fast checking
    # conflict_masks[x] is a list of bitmasks. 
    # If x is added to S, and (mask & S_mask) == mask, a 4-AP is formed.
    # The mask contains the other 3 elements of the AP.
    conflict_masks = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            # For each element in the AP, the other 3 form the conflict mask
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_masks[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_masks[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_masks[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_masks[term4].append(mask_d)
            
    # Convert to tuples for faster iteration
    conflict_masks = [tuple(lst) for lst in conflict_masks]
    
    # Bit positions for quick membership check
    bit_positions = [0] + [1 << (i - 1) for i in range(1, n + 1)]
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_masks[x]:
            if (mask & S_mask) == mask:
                return False
        return True

    def greedy_fill(S_mask, S_list, order):
        """Try to add all possible elements in the given order."""
        added_any = False
        for x in order:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
                    added_any = True
        return S_mask, S_list, added_any

    best_mask = 0
    best_size = 0
    all_elements = list(range(1, n + 1))
    
    # Run multiple restarts to find a good solution
    # Each restart: Greedy (starting from core) + Local Search with Simulated Annealing
    while time.time() - start_time < time_limit:
        # 1. Greedy Construction starting from core
        S_mask = 0
        for x in core:
            S_mask |= bit_positions[x]
        S_list = list(core)
        
        # Candidates are the rest of the numbers
        candidates = [i for i in range(1, n + 1) if i not in core]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_mask):
                S_mask |= bit_positions[x]
                S_list.append(x)
        
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            
        # 2. Simulated Annealing Local Search
        # We will try to improve the set by swaps and additions.
        
        temperature = 10.0
        min_temperature = 0.01
        cooling_rate = 0.995
        no_improvement_count = 0
        max_no_improvement = 30000
        best_in_sa_size = current_size
        best_in_sa_mask = S_mask
        
        while time.time() - start_time < time_limit and no_improvement_count < max_no_improvement:
            # Try to add an element (always accept if valid)
            added = False
            for _ in range(5):
                x = random.randint(1, n)
                if not (S_mask & bit_positions[x]):
                    if can_add(x, S_mask):
                        S_mask |= bit_positions[x]
                        S_list.append(x)
                        current_size += 1
                        added = True
                        no_improvement_count = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
                        if current_size > best_in_sa_size:
                            best_in_sa_size = current_size
                            best_in_sa_mask = S_mask
                        break
            
            if added:
                # Cool down slightly on successful add
                temperature = max(min_temperature, temperature * 0.999)
                continue
                
            # If no element could be added, try a swap
            # Pick random y in S to remove
            y_idx = random.randint(0, len(S_list) - 1)
            y = S_list[y_idx]
            
            # Pick random x not in S to add
            swap_made = False
            for _ in range(10):
                x = random.randint(1, n)
                if S_mask & bit_positions[x]:
                    continue
                
                # Check if x can be added to S \ {y}
                S_mask_without_y = S_mask & ~bit_positions[y]
                if can_add(x, S_mask_without_y):
                    # Perform swap
                    S_mask = S_mask_without_y | bit_positions[x]
                    S_list[y_idx] = x
                    swap_made = True
                    no_improvement_count = 0
                    # Swaps don't change size, so no improvement in size,
                    # but they might enable future additions.
                    # We don't update best_size here because size didn't change.
                    break
            
            if not swap_made:
                no_improvement_count += 1
                
            # Cool down
            temperature = max(min_temperature, temperature * cooling_rate)
            
            # Periodic fill to recover from any potential "holes"
            # Since swaps maintain size, and we only add if valid, the set remains valid.
            # However, a swap might open up space for *other* elements.
            if no_improvement_count % 500 == 0:
                S_mask, S_list, _ = greedy_fill(S_mask, S_list, all_elements)
                new_size = len(S_list)
                if new_size > current_size:
                    current_size = new_size
                    no_improvement_count = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = S_mask
                    if current_size > best_in_sa_size:
                        best_in_sa_size = current_size
                        best_in_sa_mask = S_mask

        # Final fill for this restart
        S_mask, S_list, _ = greedy_fill(S_mask, S_list, all_elements)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask

    # Final polish: try to add any remaining elements to the best solution
    # Reconstruct list from best_mask
    final_list = []
    for i in range(1, n + 1):
        if best_mask & bit_positions[i]:
            final_list.append(i)
    
    if not final_list:
        return [1, 2]
        
    # One last attempt to fill
    final_mask = best_mask
    for x in range(1, n + 1):
        if not (final_mask & bit_positions[x]):
            if can_add(x, final_mask):
                final_mask |= bit_positions[x]
                final_list.append(x)
    
    if not final_list:
        return [1, 2]
        
    return final_list
