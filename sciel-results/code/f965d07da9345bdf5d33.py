import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0
    
    best_size = N + 1
    best_set = []
    
    for iter in range(1000): # Increased iterations
        if time.time() > deadline:
            break
            
        random.seed(iter)
        
        S_mask = 0
        sums_mask = 0
        available_mask = ((1 << (N + 1)) - 1) ^ 1
        
        S_list = []
        
        while available_mask:
            # Get list of available elements
            avail_list = []
            temp_mask = available_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                z = lsb.bit_length() - 1
                avail_list.append(z)
                temp_mask ^= lsb
            
            if not avail_list:
                break

            random.shuffle(avail_list)
            
            best_z = -1
            max_score = -1
            
            # Pre-extract S elements
            S_elems = []
            temp_S = S_mask
            while temp_S:
                lsb_S = temp_S & -temp_S
                a = lsb_S.bit_length() - 1
                S_elems.append(a)
                temp_S ^= lsb_S
            
            for z in avail_list:
                # Check validity
                if (sums_mask >> (2*z)) & 1:
                    continue
                if (sums_mask >> z) & S_mask:
                    continue
                
                # Calculate blocked count (heuristic)
                # We want to pick z that blocks many available elements
                blocked_count = 0
                # Fast check: how many w in available_mask are blocked by z?
                # We can approximate or compute exactly if fast enough.
                # For now, use the previous logic but ensure it's correct.
                
                # Re-compute blocked candidates for z
                blocked_candidates = set()
                for a in S_elems:
                    s1 = z + a
                    if s1 % 2 == 0:
                        w1 = s1 // 2
                        if 1 <= w1 <= N and w1 != z:
                            blocked_candidates.add(w1)
                            
                    w2 = 2*z - a
                    if 1 <= w2 <= N and w2 != z:
                        blocked_candidates.add(w2)
                        
                    for a_prime in S_elems:
                        w3 = z + a - a_prime
                        if 1 <= w3 <= N and w3 != z:
                            blocked_candidates.add(w3)
                
                for w in blocked_candidates:
                    if (available_mask >> w) & 1:
                        blocked_count += 1
                
                score = blocked_count + random.random() * 0.1
                if score > max_score:
                    max_score = score
                    best_z = z
            
            if best_z == -1:
                break
            
            # Commit best_z
            S_mask |= (1 << best_z)
            S_list.append(best_z)
            
            # Update sums_mask
            sums_mask |= (1 << (2*best_z))
            for a in S_elems:
                sums_mask |= (1 << (best_z + a))
            
            # Update available_mask
            available_mask &= ~(1 << best_z)
            
            # Remove blocked elements
            for w in blocked_candidates:
                if (available_mask >> w) & 1:
                    available_mask &= ~(1 << w)
            
        # --- Repair Step ---
        # Check if S_list is maximal
        # If not, add missing elements
        current_set = set(S_list)
        # Rebuild sums for current_set
        curr_sums = set()
        for a in current_set:
            for b in current_set:
                if a <= b:
                    curr_sums.add(a + b)
        
        # Check all elements in [1, N]
        for x in range(1, N + 1):
            if x in current_set:
                continue
            
            # Check if x can be added
            valid = True
            # 2x not in curr_sums
            if 2*x in curr_sums:
                valid = False
                break
            # x + a not in curr_sums for any a in current_set
            for a in current_set:
                if x + a in curr_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            # x is addable, so set is not maximal. Add x.
            current_set.add(x)
            # Update curr_sums
            curr_sums.add(2*x)
            for a in current_set:
                if a != x: # Avoid double adding if we iterate over the new set? 
                    # Actually, we just added x. We need to add x+a for all a in current_set (including x? 2x already added).
                    # So x+a for a in current_set \ {x}
                    curr_sums.add(x + a)
            # Note: We don't need to re-check other elements immediately, 
            # but adding x might make other elements invalid? 
            # No, if an element y was invalid before, it remains invalid.
            # If y was valid before, it might become invalid now? 
            # If y was valid, it didn't collide with old sums.
            # New sums are 2x and x+a.
            # y might collide with 2x or x+a.
            # So y might become invalid.
            # This is fine, we just want a maximal set.
            # We continue scanning.
        
        # Convert to list
        final_list = list(current_set)
        
        if len(final_list) < best_size:
            best_size = len(final_list)
            best_set = final_list[:]
            
    return best_set
