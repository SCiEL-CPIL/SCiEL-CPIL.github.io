import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85  # Safety margin before 90s kill
    
    # Precompute conflict masks for divisibility
    # conflict_mask[u] has bit v set if u and v are distinct and one divides the other
    MAX_SUM = 8000
    conflict_mask = [0] * (MAX_SUM + 1)
    
    # For each divisor d, it conflicts with all its multiples m > d
    for d in range(1, MAX_SUM + 1):
        m = d * 2
        while m <= MAX_SUM:
            conflict_mask[d] |= (1 << m)
            conflict_mask[m] |= (1 << d)
            m += d
            
    def is_valid_bitmask(S):
        """
        Check if a set S is valid.
        S is a list of ints.
        Returns True if valid.
        """
        if not S:
            return True
        
        # Generate subset sums as a bitmask
        sums_mask = 1  # bit 0 set (sum 0)
        
        for x in S:
            # new_sums_mask = sums_mask | (sums_mask << x)
            # But we need to check for duplicate sums first.
            # If (sums_mask & (sums_mask << x)) != 0, then there is a collision.
            shifted = sums_mask << x
            if (sums_mask & shifted) != 0:
                return False
            sums_mask |= shifted
        
        # Now check divisibility conflicts among non-zero sums
        # We need to check if any two distinct non-zero sums in sums_mask conflict.
        # We can iterate through the bits set in sums_mask (excluding 0)
        # and check against the rest.
        
        # Optimization: 
        # For each sum u in sums_mask, check if (sums_mask & conflict_mask[u]) != 0.
        # However, this is O(N_sums * N_conflicts).
        # A faster way: 
        # Iterate through set bits in sums_mask.
        # For each u, we know all v that conflict with u are in conflict_mask[u].
        # We just need to know if any such v is also in sums_mask.
        # So: if (sums_mask & conflict_mask[u]) != 0, it's invalid.
        # We must be careful not to count u itself if u was in conflict_mask[u] (it's not).
        
        m = sums_mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            if u > 0:
                if (sums_mask & conflict_mask[u]) != 0:
                    return False
            m ^= lsb
            
        return True

    def try_add_fast(x, sums_mask):
        """
        Try to add x to a set represented by sums_mask.
        Returns (True, new_sums_mask) if valid, (False, None) otherwise.
        """
        # Check for sum collision
        shifted = sums_mask << x
        if (sums_mask & shifted) != 0:
            return (False, None)
        
        new_sums_mask = sums_mask | shifted
        
        # Check divisibility conflicts for the NEW sums
        # The new sums are exactly the bits in `shifted`.
        # For each new sum u in `shifted`, check if it conflicts with ANY sum in `new_sums_mask`.
        # Note: conflict_mask[u] contains all v that conflict with u.
        # So we check if (new_sums_mask & conflict_mask[u]) != 0.
        
        m = shifted
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            # Check if u conflicts with any sum in the new set
            if (new_sums_mask & conflict_mask[u]) != 0:
                return (False, None)
            m ^= lsb
            
        return (True, new_sums_mask)

    def greedy_build(candidates):
        """Greedy build: iterate through candidates, add if valid."""
        S = []
        sums_mask = 1 # {0}
        
        for x in candidates:
            ok, new_mask = try_add_fast(x, sums_mask)
            if ok:
                S.append(x)
                sums_mask = new_mask
                
        return S, sums_mask

    def local_search(start_S, start_mask):
        """
        Local search: try to swap elements to increase size.
        """
        S = list(start_S)
        sums_mask = start_mask
        improved = True
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # Try to add a new element if possible? 
            # The greedy builder should have added all possible elements in the candidate order.
            # But local search can try to swap to allow a larger set.
            
            # Strategy: Remove one element, try to add two? Or just swap?
            # Swapping keeps size constant. We want to increase size.
            # So we should try to add an element. If we can't, maybe removing one allows adding two?
            
            # Let's try a simple hill climb: 
            # 1. Try to add any number 1..256 not in S.
            # 2. If stuck, try swapping S[i] with x not in S, then try to add y.
            
            # First, try simple addition
            current_mask = sums_mask
            for x in range(1, 257):
                if x in S:
                    continue
                ok, new_mask = try_add_fast(x, current_mask)
                if ok:
                    S.append(x)
                    sums_mask = new_mask
                    improved = True
                    # Restart the loop to try adding more
                    break
            
            if improved:
                continue
                
            # If no simple addition, try swaps
            # This is expensive. Only do if we have time.
            if time.time() - start_time > time_limit - 5:
                break
                
            for i in range(len(S)):
                if time.time() - start_time > time_limit - 5:
                    break
                    
                removed = S[i]
                S_reduced = S[:i] + S[i+1:]
                
                # Recompute mask for reduced set
                # This is O(N^2) if done naively. 
                # To speed up, we could maintain masks, but recomputing is safer for correctness in a quick edit.
                # With N <= 10, recomputing is fast.
                mask_reduced = 1
                for x in S_reduced:
                    mask_reduced |= (mask_reduced << x)
                    # Note: we assume S_reduced is valid, so no collisions/divisibility issues internally
                    # but we must check collisions during generation? No, S_reduced is subset of valid S, so it's valid.
                    # Wait, is a subset of a valid set always valid?
                    # Yes, if no two subset sums of S divide each other, then no two subset sums of S_reduced do.
                    # And distinctness holds.
                
                # Try to add a replacement x
                found_swap = False
                for x in range(1, 257):
                    if x in S_reduced:
                        continue
                    ok, new_mask_reduced = try_add_fast(x, mask_reduced)
                    if ok:
                        # Now try to add a second element y to the new set?
                        # That would increase size by 1.
                        # Let's just do the swap first, then try to add.
                        new_S = S_reduced + [x]
                        
                        # Try to add any y
                        for y in range(1, 257):
                            if y in new_S:
                                continue
                            ok2, final_mask = try_add_fast(y, new_mask_reduced)
                            if ok2:
                                S = new_S + [y]
                                sums_mask = final_mask
                                improved = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                        
                        # If we can't add a second, maybe just swap? 
                        # But swap doesn't increase size. We want to maximize size.
                        # So only accept if we can grow.
                
                if found_swap:
                    break

        return S, sums_mask

    # Strategy 1: Binary signature / Unique Highest Bit Greedy
    # Group numbers by their highest set bit
    by_highest_bit = {}
    for n in range(1, 257):
        hb = n.bit_length() - 1
        if hb not in by_highest_bit:
            by_highest_bit[hb] = []
        by_highest_bit[hb].append(n)
    
    best_local = []
    
    # Try multiple randomizations of the bit groups
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Shuffle the order of bit groups
        bit_keys = sorted(by_highest_bit.keys(), reverse=True)
        random.shuffle(bit_keys)
        
        S, sums_mask = greedy_build([]) # Init empty
        
        for bk in bit_keys:
            if time.time() - start_time > time_limit:
                break
            candidates = by_highest_bit[bk][:]
            random.shuffle(candidates)
            # Greedy add from these candidates
            for x in candidates:
                ok, new_mask = try_add_fast(x, sums_mask)
                if ok:
                    S.append(x)
                    sums_mask = new_mask
                    break # Take at most one from this bit group to ensure unique highest bits?
                    # Actually, taking more than one from the same highest bit group might be valid 
                    # if the lower bits are distinct enough, but it increases collision risk.
                    # The "binary signature" idea implies one per bit to guarantee distinctness easily.
                    # Let's stick to one per bit group for the "signature" strategy.
        
        if len(S) > len(best_local):
            best_local = S[:]
            # Try local search to extend
            if time.time() - start_time < time_limit - 2:
                ls_S, ls_mask = local_search(best_local, sums_mask)
                if len(ls_S) > len(best_local):
                    best_local = ls_S[:]

    if len(best_local) > 0:
        best_size = len(best_local)
        best_set = best_local[:]
    else:
        best_size = 0
        best_set = []

    # Strategy 2: Standard Randomized Greedy (no bit grouping constraint)
    # This might find denser sets if the bit grouping is too restrictive
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
        
        candidates = list(range(1, 257))
        # Bias towards larger numbers? Larger numbers have fewer multiples in range, 
        # so they are "safer" against divisibility constraints with other large sums.
        # But small numbers are "safer" against collision?
        # Let's try a mix.
        
        if attempt % 2 == 0:
            random.shuffle(candidates)
        else:
            candidates.sort(reverse=True)
            random.shuffle(candidates) # Shuffle to break ties? No, reverse sort is deterministic.
            # Let's just shuffle all.
            random.shuffle(candidates)
            
        S, sums_mask = greedy_build(candidates)
        
        if len(S) > len(best_local):
            best_local = S[:]
            if time.time() - start_time < time_limit - 2:
                ls_S, ls_mask = local_search(best_local, sums_mask)
                if len(ls_S) > len(best_local):
                    best_local = ls_S[:]

    if len(best_local) > best_size:
        best_size = len(best_local)
        best_set = best_local[:]

    # Final verification
    if best_set:
        if not is_valid_bitmask(best_set):
            best_set = [31, 32, 34]
            best_size = 3

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
