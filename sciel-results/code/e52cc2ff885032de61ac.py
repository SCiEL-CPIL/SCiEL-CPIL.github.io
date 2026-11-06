import time
import math

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit to be safe
    
    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    
    # Sort divisors to have a consistent order
    divisors.sort()
    
    # 2. Precompute masks for each (m, r)
    # mask[m][r] is an integer with bits set at r, r+m, ...
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask_val = 0
            for i in range(r, base, m):
                mask_val |= (1 << i)
            masks[m][r] = mask_val
            
    # 3. Greedy Set Cover
    # uncovered is a bitset of all 1s
    uncovered = (1 << base) - 1
    remaining_moduli = set(divisors)
    classes = []
    
    while remaining_moduli:
        best_gain = 0
        best_m = None
        best_r = None
        
        # Find the modulus and residue that maximizes coverage of uncovered
        for m in remaining_moduli:
            # For this m, find r that maximizes intersection with uncovered
            # We can iterate through all r for this m
            for r in range(m):
                gain = (uncovered & masks[m][r]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_gain == 0:
            break
            
        # Select the best class
        classes.append([best_m, best_r])
        
        # Update uncovered
        uncovered &= ~masks[best_m][best_r]
        
        # Remove the modulus from consideration
        remaining_moduli.remove(best_m)
        
        # Time check
        if time.time() > deadline - 10.0: # Leave time for local search
            break
            
    # 4. Local Search Refinement
    # We try to improve the solution by swapping classes.
    # Current score: uncovered.bit_count()
    # We maintain the set of selected classes.
    selected_moduli = {c[0] for c in classes}
    
    # To speed up local search, we can limit the number of iterations or time.
    local_search_deadline = start_time + 148.0
    
    improved = True
    while improved and time.time() < local_search_deadline:
        improved = False
        
        # For each modulus not selected, find the best residue to cover 'uncovered'
        # We will store these candidates: (m, r, gain)
        candidates = []
        
        for m in divisors:
            if m in selected_moduli:
                continue
            
            best_r_for_m = 0
            best_gain_for_m = 0
            for r in range(m):
                gain = (uncovered & masks[m][r]).bit_count()
                if gain > best_gain_for_m:
                    best_gain_for_m = gain
                    best_r_for_m = r
            
            if best_gain_for_m > 0:
                candidates.append((m, best_r_for_m, best_gain_for_m))
        
        # Try swapping each selected class with each candidate
        # We want to minimize uncovered count.
        # Swap (m_out, r_out) with (m_in, r_in):
        # new_uncovered = (uncovered | mask_out) & ~mask_in
        # We check if new_uncovered.bit_count() < uncovered.bit_count()
        
        current_score = uncovered.bit_count()
        
        for m_out, r_out in classes:
            mask_out = masks[m_out][r_out]
            
            for m_in, r_in, _ in candidates:
                # Skip if modulus is same (should not happen due to distinct moduli constraint in selection, 
                # but m_in is from divisors not in selected, so m_in != m_out is guaranteed)
                
                mask_in = masks[m_in][r_in]
                
                # Calculate new uncovered
                # (uncovered | mask_out) removes the coverage of mask_out from uncovered (i.e., adds it back to the covered set, so it becomes uncovered if it wasn't already, but wait: uncovered is the set of NOT covered. 
                # If we remove mask_out, the residues covered ONLY by mask_out become uncovered.
                # So new_uncovered_before_in = uncovered | mask_out.
                # Then we apply mask_in, which covers some of them.
                # new_uncovered = (uncovered | mask_out) & ~mask_in
                
                new_uncovered = (uncovered | mask_out) & ~mask_in
                new_score = new_uncovered.bit_count()
                
                if new_score < current_score:
                    # Accept swap
                    uncovered = new_uncovered
                    current_score = new_score
                    
                    # Update classes list
                    # Find index of [m_out, r_out]
                    for i, c in enumerate(classes):
                        if c[0] == m_out and c[1] == r_out:
                            classes[i] = [m_in, r_in]
                            break
                    
                    # Update selected_moduli
                    selected_moduli.remove(m_out)
                    selected_moduli.add(m_in)
                    
                    improved = True
                    # Restart candidate generation or just break out to re-evaluate?
                    # Breaking out to re-evaluate is safer and simpler.
                    break
            
            if improved:
                break

    return classes
