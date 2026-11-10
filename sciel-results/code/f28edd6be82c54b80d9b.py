import time
import random

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_size = N + 1
    best_set = []
    
    # Precompute: for each element w, the "forbidden sum mask" that would block w
    # w is blocked if: (sums_mask & block_mask[w]) != 0
    # where block_mask[w] has bit 2w set, and bits (w+a) for all a that could be in S
    # Actually: w is blocked if 2w is in sums, OR if exists a in S such that a+w is in sums
    # The second condition: (sums_mask >> w) & S_mask != 0
    # This is hard to precompute as a single mask since it depends on S_mask.
    # But we can check: for each w, it's blocked if (sums_mask >> (2*w)) & 1 or ((sums_mask >> w) & S_mask)
    
    def greedy_extend_fast(S_mask, sums_mask, available_mask, rng):
        """
        Fast greedy extension using bit operations.
        """
        S_list = []
        m = S_mask
        while m:
            lsb = m & -m
            S_list.append(lsb.bit_length() - 1)
            m ^= lsb
        
        while available_mask:
            avail_list = []
            m = available_mask
            while m:
                lsb = m & -m
                avail_list.append(lsb.bit_length() - 1)
                m ^= lsb
            
            if not avail_list:
                break
            
            rng.shuffle(avail_list)
            
            best_z = -1
            max_blocked = -1
            best_score = -1.0
            
            for z in avail_list:
                # Check if z is valid to add to current S
                # 1. 2z not in sums
                if (sums_mask >> (2*z)) & 1:
                    continue
                # 2. No collision: (sums >> z) & S_mask == 0
                if (sums_mask >> z) & S_mask:
                    continue
                
                # Simulate adding z
                new_S_mask = S_mask | (1 << z)
                new_sums_mask = sums_mask | (1 << (2*z))
                
                # Add z+a for a in S
                temp_S = S_mask
                add_bits = 0
                while temp_S:
                    lsb_S = temp_S & -temp_S
                    a = lsb_S.bit_length() - 1
                    add_bits |= (1 << (z + a))
                    temp_S ^= lsb_S
                new_sums_mask |= add_bits
                
                # Count how many other available elements are blocked
                # Use bit operations: for each w in avail, check if blocked by new_sums_mask and new_S_mask
                # blocked if: (new_sums_mask >> (2*w)) & 1 OR ((new_sums_mask >> w) & new_S_mask)
                # 
                # Vectorized approach:
                # For condition 1: create a mask of all 2w for w in avail, AND with new_sums_mask
                # For condition 2: this is harder to vectorize directly.
                # 
                # Let's use a hybrid: iterate avail but use fast bit checks
                blocked_count = 0
                for w in avail_list:
                    if w == z:
                        continue
                    if (new_sums_mask >> (2*w)) & 1:
                        blocked_count += 1
                        continue
                    if (new_sums_mask >> w) & new_S_mask:
                        blocked_count += 1
                
                score = blocked_count + rng.random() * 0.5
                
                if score > best_score:
                    best_score = score
                    best_z = z
            
            if best_z == -1:
                break
            
            # Commit best_z
            S_mask = S_mask | (1 << best_z)
            S_list.append(best_z)
            
            # Update sums_mask
            sums_mask |= (1 << (2*best_z))
            temp_S = S_mask ^ (1 << best_z)
            while temp_S:
                lsb_S = temp_S & -temp_S
                a = lsb_S.bit_length() - 1
                sums_mask |= (1 << (best_z + a))
                temp_S ^= lsb_S
            
            # Update available_mask - remove blocked elements
            available_mask &= ~(1 << best_z)
            
            # Remove blocked elements using bit operations
            new_avail_mask = 0
            temp_avail = available_mask
            while temp_avail:
                lsb_w = temp_avail & -temp_avail
                w = lsb_w.bit_length() - 1
                if not ((sums_mask >> (2*w)) & 1):
                    if not ((sums_mask >> w) & S_mask):
                        new_avail_mask |= lsb_w
                temp_avail ^= lsb_w
            
            available_mask = new_avail_mask
        
        return S_mask, sums_mask, available_mask, S_list
    
    def compute_sums_mask(S_list):
        """Compute sums mask for a Sidon set given as list."""
        sums_mask = 0
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                sums_mask |= (1 << (a + S_list[j]))
        return sums_mask
    
    def compute_avail_mask(S_mask, sums_mask):
        """Compute available mask for given S and sums."""
        avail = 0
        for w in range(1, N + 1):
            if (S_mask >> w) & 1:
                continue
            if not ((sums_mask >> (2*w)) & 1):
                if not ((sums_mask >> w) & S_mask):
                    avail |= (1 << w)
        return avail
    
    # Main Loop
    for iter in range(200):
        if time.time() > deadline:
            break
            
        rng = random.Random(iter * 7919 + 13)
        
        S_mask = 0
        sums_mask = 0
        available_mask = ((1 << (N + 1)) - 1) ^ 1
        
        S_mask, sums_mask, available_mask, S_list = greedy_extend_fast(S_mask, sums_mask, available_mask, rng)
        current_size = len(S_list)
        
        # Local Search: Try to remove elements and re-extend
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list_copy = S_list[:]
            rng.shuffle(S_list_copy)
            
            for x in S_list_copy:
                if time.time() > deadline:
                    break
                
                new_S_mask = S_mask ^ (1 << x)
                new_S_list = [s for s in S_list if s != x]
                new_sums_mask = compute_sums_mask(new_S_list)
                new_avail_mask = compute_avail_mask(new_S_mask, new_sums_mask)
                
                new_S_mask_ext, new_sums_mask_ext, _, new_S_list_ext = greedy_extend_fast(
                    new_S_mask, new_sums_mask, new_avail_mask, rng
                )
                
                if len(new_S_list_ext) < current_size:
                    S_mask = new_S_mask_ext
                    sums_mask = new_sums_mask_ext
                    S_list = new_S_list_ext
                    current_size = len(S_list)
                    improved = True
                    break
        
        if current_size < best_size:
            best_size = current_size
            best_set = S_list[:]
    
    # Final polish: try removing each element from best_set and re-extending
    if best_set and time.time() < deadline:
        best_S_mask = 0
        for x in best_set:
            best_S_mask |= (1 << x)
        best_sums_mask = compute_sums_mask(best_set)
        
        polished = True
        while polished and time.time() < deadline:
            polished = False
            best_list_copy = best_set[:]
            random.Random(42).shuffle(best_list_copy)
            
            for x in best_list_copy:
                if time.time() > deadline:
                    break
                
                new_S_mask = best_S_mask ^ (1 << x)
                new_S_list = [s for s in best_set if s != x]
                new_sums_mask = compute_sums_mask(new_S_list)
                new_avail_mask = compute_avail_mask(new_S_mask, new_sums_mask)
                
                rng2 = random.Random(hash((x, tuple(sorted(best_set)))) % (2**31))
                new_S_mask_ext, new_sums_mask_ext, _, new_S_list_ext = greedy_extend_fast(
                    new_S_mask, new_sums_mask, new_avail_mask, rng2
                )
                
                if len(new_S_list_ext) < len(best_set):
                    best_set = new_S_list_ext
                    best_S_mask = new_S_mask_ext
                    best_sums_mask = new_sums_mask_ext
                    polished = True
                    break
    
    return best_set
