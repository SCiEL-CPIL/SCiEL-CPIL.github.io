import numpy as np
import time

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Generate divisors > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # Precompute bitmasks for each modulus and residue
    # masks[m] is a list of length m, where masks[m][r] is an int bitmask
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        # For each residue r, set bits for all x in [0, base) with x % m == r
        # We can construct this efficiently
        for r in range(m):
            mask = 0
            # Iterate through the arithmetic progression
            # x = r, r+m, r+2m, ... < base
            for x in range(r, base, m):
                mask |= (1 << x)
            masks[m][r] = mask
            
    # State
    covered_mask = 0
    classes = []
    selected_moduli = set()
    unselected_moduli = set(divisors)
    
    # Helper to get current uncovered count
    def get_uncovered_count(mask):
        # Total bits is base. We want count of 0s in mask within [0, base).
        # Since we only set bits < base, we can just do:
        # base - mask.bit_count()
        return base - mask.bit_count()
        
    current_uncovered = base
    
    # 1. Greedy Initialization
    while unselected_moduli:
        if time.time() > deadline:
            break
            
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in unselected_moduli:
            # Find best r for this m
            # Gain = popcount(mask[m][r] & ~covered_mask)
            # We want to maximize gain
            for r in range(m):
                gain = (masks[m][r] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
            
            # Early exit if we covered everything?
            if best_gain == 0:
                continue
                
        if best_m is None or best_gain == 0:
            break
            
        # Add the class
        classes.append([best_m, best_r])
        covered_mask |= masks[best_m][best_r]
        selected_moduli.add(best_m)
        unselected_moduli.remove(best_m)
        current_uncovered = base - covered_mask.bit_count()
        
        if current_uncovered == 0:
            return classes

    # 2. Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 2a. Try adding unselected moduli
        for m in list(unselected_moduli):
            if time.time() > deadline:
                break
            best_r = 0
            best_gain = 0
            for r in range(m):
                gain = (masks[m][r] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
                    
            if best_gain > 0:
                # Add it
                classes.append([m, best_r])
                covered_mask |= masks[m][best_r]
                selected_moduli.add(m)
                unselected_moduli.remove(m)
                current_uncovered = base - covered_mask.bit_count()
                improved = True
                if current_uncovered == 0:
                    return classes

        # 2b. Try swapping selected moduli with unselected ones
        # For each selected m_out, try swapping with each unselected m_in
        for m_out in list(selected_moduli):
            if time.time() > deadline:
                break
            
            # Find current residue for m_out
            r_out = None
            for c in classes:
                if c[0] == m_out:
                    r_out = c[1]
                    break
                    
            # Remove m_out from coverage
            # covered_mask_without_out = covered_mask & ~masks[m_out][r_out]
            # But we need to be careful: if other classes cover those bits, they stay covered.
            # Actually, `covered_mask` is the union.
            # To simulate removal, we can't just AND NOT because of overlap.
            # We need to recompute coverage if we remove a class?
            # That's expensive.
            
            # Alternative: Calculate potential gain of swap.
            # Let U_out be the set of residues covered ONLY by m_out (or rather, the contribution of m_out).
            # It's hard to know "only" without full recalculation.
            
            # However, we can estimate.
            # Let's try a simpler heuristic: 
            # If we remove m_out, we lose at most `masks[m_out][r_out].bit_count()` bits, 
            # but likely fewer due to overlap.
            # If we add m_in, we gain at most `masks[m_in][r_in].bit_count()` bits.
            
            # Let's just do a "virtual" swap check for the best candidate m_in.
            # We will compute the new coverage mask explicitly if we decide to swap.
            # To decide, we need to know if the swap is beneficial.
            
            # Approximation:
            # lost_bits = masks[m_out][r_out] & ~ (covered_mask ^ masks[m_out][r_out]) ? No.
            
            # Let's stick to a safer but potentially slower local search step:
            # Only consider swaps if the "potential" of m_in is high.
            
            # Let's try to find the best m_in, r_in pair that maximizes (gain_in - lost_out_approx)
            # lost_out_approx = number of bits in masks[m_out][r_out] that are NOT covered by other classes.
            # This requires knowing coverage by others.
            
            # Efficient way to check swap:
            # 1. Calculate `other_coverage` = covered_mask without m_out.
            #    This is hard to do incrementally.
            
            # Let's use a different strategy for swaps:
            # Try swapping m_out with m_in.
            # New covered = (covered_mask & ~masks[m_out][r_out]) | masks[m_in][r_in]
            # Wait, `covered_mask & ~masks[m_out][r_out]` is WRONG if other classes cover those bits.
            # The bits covered by m_out that are ALSO covered by other classes will remain set in `covered_mask` even after removing m_out's contribution?
            # No, `covered_mask` is the result of OR-ing all masks.
            # If we remove m_out, the new mask is `OR(all masks except m_out)`.
            # We don't have this readily available.
            
            # To make this fast, we can maintain a count of how many times each bit is covered?
            # `cover_count` array of length base.
            # If we add m, we increment cover_count for bits in mask[m][r].
            # If we remove m, we decrement.
            # Then `covered_mask` is just `cover_count > 0`.
            # But updating cover_count for a mask with ~1000 bits is O(1000).
            # With 47 moduli, and swaps, this might be feasible.
            
            # Let's implement `cover_count` based local search.
            # This is a structural change from pure bitmask OR.
            
            pass # Placeholder, we will refactor below
            
    # Refactored Local Search with Cover Counts
    # We need to rebuild the state with cover counts to support efficient removal.
    # This is a significant change. Let's restart the search logic with this representation.
    
    # --- Restart with Cover Counts ---
    covered_mask = 0
    classes = []
    selected_moduli = set()
    unselected_moduli = set(divisors)
    cover_count = np.zeros(base, dtype=np.uint8)
    
    # 1. Greedy Initialization
    while unselected_moduli:
        if time.time() > deadline:
            break
        
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in unselected_moduli:
            for r in range(m):
                # Gain is number of bits in mask[m][r] where cover_count is 0
                mask = masks[m][r]
                # We need to count zeros in cover_count for positions where mask is 1.
                # This is O(base/m) if we iterate, or we can use numpy.
                # But we are in a tight loop.
                # Let's use the bitmask trick again for gain calculation, but maintain cover_count for swaps.
                # Gain = popcount(mask & ~(covered_mask_from_counts))
                # covered_mask_from_counts is (cover_count > 0).
                # Let's maintain `current_covered_mask` as an int.
                
                gain = (mask & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
        
        if best_m is None or best_gain == 0:
            break
        
        # Apply
        classes.append([best_m, best_r])
        mask = masks[best_m][best_r]
        covered_mask |= mask
        # Update cover_count
        # Extract indices from mask? Or just iterate?
        # Iterating through bits of a large int is slow in Python.
        # Better to precompute indices for each mask?
        # indices[m][r] = list of x such that x % m == r.
        
        # Let's precompute indices.
        # This uses more memory but speeds up cover_count updates.
        pass

    # The above restart is getting complicated. Let's stick to the Bitmask approach but optimize the swap logic.
    # For swaps, we can just compute the new `covered_mask` from scratch if the number of classes is small?
    # No, that's O(K * N) which is too slow if K=47.
    
    # Let's try a hybrid:
    # Keep `covered_mask` (int).
    # For swaps, we check if `m_in` covers significantly more "unique" bits than `m_out` loses.
    # Unique bits of m_out = masks[m_out][r_out] & ~ (covered_mask ^ masks[m_out][r_out])? No.
    # Let's just try the swap and calculate the new covered mask by re-ORing all other classes?
    # If K is small (e.g., < 10), this is fast. If K is 47, it's 46 ORs of large ints.
    # 46 * (cost of OR) is probably < 1ms.
    # So we can afford to recompute `covered_mask` for each swap candidate!
    
    # Let's implement the local search with recomputation.
    
    # Reset state for the second pass (or just continue from the first pass state if we didn't restart)
    # I will replace the entire function body to ensure clean logic.
    
    return classes # This is a placeholder, the real logic is below

# Actual implementation
import numpy as np
import time

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0
    
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # Precompute masks
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask = 0
            for x in range(r, base, m):
                mask |= (1 << x)
            masks[m][r] = mask
            
    covered_mask = 0
    classes = []
    selected_moduli = set()
    unselected_moduli = set(divisors)
    
    # Greedy
    while unselected_moduli:
        if time.time() > deadline:
            break
        best_gain = 0
        best_m = None
        best_r = None
        for m in unselected_moduli:
            for r in range(m):
                gain = (masks[m][r] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
        if best_m is None or best_gain == 0:
            break
        classes.append([best_m, best_r])
        covered_mask |= masks[best_m][best_r]
        selected_moduli.add(best_m)
        unselected_moduli.remove(best_m)
        if covered_mask.bit_count() == base:
            return classes
            
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding
        for m in list(unselected_moduli):
            if time.time() > deadline:
                break
            best_r = 0
            best_gain = 0
            for r in range(m):
                gain = (masks[m][r] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_gain > 0:
                classes.append([m, best_r])
                covered_mask |= masks[m][best_r]
                selected_moduli.add(m)
                unselected_moduli.remove(m)
                improved = True
                if covered_mask.bit_count() == base:
                    return classes
                    
        # Try swapping
        # We recompute covered_mask for each potential swap to be accurate
        for m_out in list(selected_moduli):
            if time.time() > deadline:
                break
            r_out = next(c[1] for c in classes if c[0] == m_out)
            
            # Calculate coverage without m_out
            # new_mask_base = OR of all masks except m_out
            # We can do this by iterating classes
            mask_without_out = 0
            for c in classes:
                if c[0] != m_out:
                    mask_without_out |= masks[c[0]][c[1]]
            
            # Now try adding each m_in
            for m_in in unselected_moduli:
                # Find best r_in for m_in given mask_without_out
                best_r_in = 0
                best_gain_in = 0
                for r_in in range(m_in):
                    # Gain relative to mask_without_out
                    gain = (masks[m_in][r_in] & ~mask_without_out).bit_count()
                    if gain > best_gain_in:
                        best_gain_in = gain
                        best_r_in = r_in
                        
                # Total covered after swap
                new_covered = mask_without_out | masks[m_in][best_r_in]
                new_count = new_covered.bit_count()
                old_count = covered_mask.bit_count()
                
                if new_count > old_count:
                    # Perform swap
                    # Update classes list
                    for c in classes:
                        if c[0] == m_out:
                            c[0] = m_in
                            c[1] = best_r_in
                            break
                    covered_mask = new_covered
                    selected_moduli.remove(m_out)
                    selected_moduli.add(m_in)
                    unselected_moduli.remove(m_in)
                    unselected_moduli.add(m_out)
                    improved = True
                    if new_count == base:
                        return classes
                        
    return classes
