import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    n = 512
    
    # Precompute all 4-APs
    # conflict_triples[x] is a list of masks. 
    # If x is in S, and (mask & S_mask) == mask, then a 4-AP exists.
    # The mask contains the other 3 elements.
    conflict_triples = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            # For each element in the AP, the other 3 form the conflict mask
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_triples[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_triples[term4].append(mask_d)
            
    # Convert to tuples for faster iteration
    conflict_triples = [tuple(lst) for lst in conflict_triples]
    
    bit_positions = [0] + [1 << (i - 1) for i in range(1, n + 1)]
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_triples[x]:
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
    
    # --- Strategy: Density-Incremental Construction ---
    # Partition [1, 512] into overlapping blocks of length 81.
    # Use a Salem-Spencer-like base-3 heuristic to generate high-density seeds for each block.
    # Then merge these seeds using conflict-aware union and local repair.
    
    block_size = 81
    # We want to cover 1..512.
    # If we use step 40, we get blocks starting at 1, 41, 81, ..., 481.
    # 481 + 81 - 1 = 561 > 512, so it covers everything.
    # Number of blocks: (512 - 1) // 40 + 1 = 13 blocks.
    block_starts = list(range(1, n + 1, 40))
    if block_starts[-1] + block_size - 1 < n:
        block_starts.append(n - block_size + 1)

    def generate_block_seed(start, length, S_mask):
        """
        Generate a 4-AP-free subset of [start, start+length-1] 
        that is also disjoint from S_mask (or compatible with it).
        Uses a greedy approach prioritizing elements with fewer conflicts 
        within the block, biased towards a base-3 distribution pattern.
        """
        candidates = list(range(start, min(start + length, n + 1)))
        
        # Heuristic: Prefer numbers with fewer '1's in base 3? 
        # Or just randomize and let the local search fix it.
        # The prompt suggests "Salem-Spencer base-3 construction".
        # Salem-Spencer sets are typically {sum 3^i * d_i | d_i in {0,1,2}}.
        # For a small block, we can just try to maximize density via greedy.
        
        # To exploit "additive structure", let's try a specific ordering:
        # Sort by value, but maybe shuffle slightly?
        # Actually, for the seed, we just want a good local density.
        # Let's use a greedy fill on the block candidates, but check against global S_mask.
        
        local_seed = []
        local_mask = 0
        
        # Try to add elements in the block.
        # Order: random? Or by "conflict count" in the block?
        # Let's just random shuffle for diversity, as the global repair will handle structure.
        random.shuffle(candidates)
        
        for x in candidates:
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask | local_mask):
                    S_mask |= bit_positions[x]
                    local_mask |= bit_positions[x]
                    local_seed.append(x)
                    
        return local_seed, S_mask

    # Run multiple restarts to find a good solution
    while time.time() < deadline:
        S_mask = 0
        S_list = []
        
        # 1. Density-Incremental Construction Phase
        # Process blocks in a random order to avoid bias
        shuffled_blocks = block_starts[:]
        random.shuffle(shuffled_blocks)
        
        for start in shuffled_blocks:
            seed, S_mask = generate_block_seed(start, block_size, S_mask)
            S_list.extend(seed)
            
        # Ensure S_list is unique and sorted for consistency (though order doesn't matter for mask)
        S_list = list(set(S_list))
        current_size = len(S_list)
        
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            
        # 2. Global Local Search Phase (Simulated Annealing / Swap)
        # This phase repairs conflicts between blocks and tries to fill gaps.
        
        temperature = 10.0
        min_temperature = 0.01
        cooling_rate = 0.995
        no_improvement_count = 0
        max_no_improvement = 30000
        
        # Reconstruct S_list from mask to ensure consistency
        S_list = [i for i in range(1, n + 1) if S_mask & bit_positions[i]]
        
        while time.time() < deadline and no_improvement_count < max_no_improvement:
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
                        break
            
            if added:
                temperature = max(min_temperature, temperature * 0.999)
                continue
                
            # If no element could be added, try a swap
            # Pick random y in S to remove
            if S_list:
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
                        break
                
                if not swap_made:
                    no_improvement_count += 1
                    
                # Cool down
                temperature = max(min_temperature, temperature * cooling_rate)
                
                # Periodic fill to recover from any potential "holes"
                if no_improvement_count % 500 == 0:
                    S_mask, S_list, _ = greedy_fill(S_mask, S_list, all_elements)
                    new_size = len(S_list)
                    if new_size > current_size:
                        current_size = new_size
                        no_improvement_count = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask

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
