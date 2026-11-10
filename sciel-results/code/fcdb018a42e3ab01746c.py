import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 51975
    
    # Generate divisors
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    # Filter odd divisors > 1
    divisors = [d for d in divisors if d > 1 and d % 2 != 0]
    divisors.sort()
    
    # Precompute bitmasks for each (m, r)
    # mask[m][r] will be an integer with bits set for residues r, r+m, r+2m...
    masks = {}
    for m in divisors:
        masks[m] = []
        for r in range(m):
            mask = 0
            # Set bits for r, r+m, r+2m, ...
            for x in range(r, N, m):
                mask |= (1 << x)
            masks[m].append(mask)
            
    # Greedy Construction
    covered = 0
    selected_classes = [] # List of [m, r]
    used_moduli = set()
    
    # We prioritize smaller moduli as they cover more residues
    for m in divisors:
        if time.time() > deadline:
            break
            
        best_r = 0
        best_gain = -1
        m_masks = masks[m]
        
        for r in range(m):
            # Count bits in (m_masks[r] & ~covered)
            new_bits = m_masks[r] & ~covered
            # Python int.bit_count() is fast in 3.10+
            gain = new_bits.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_r = r
                if gain == N // m: # Maximum possible gain for this m
                    break
                    
        if best_gain > 0:
            selected_classes.append([m, best_r])
            used_moduli.add(m)
            covered |= masks[m][best_r]
        else:
            # If no gain, we might still want to include it if it doesn't hurt?
            # No, we want to maximize coverage. If gain is 0, it adds nothing.
            # But the problem says "distinct moduli". It doesn't say we MUST use all.
            # "With every divisor ... available once".
            # Score is uncovered residues.
            # Adding a class with 0 gain doesn't change score, but uses up a modulus.
            # It's better to skip it to save time, or include it if it might help later?
            # Actually, if gain is 0, it covers nothing new. It's redundant.
            # We should skip it to keep the list small and potentially allow better swaps later?
            # Or just skip it.
            pass

    # Local Search / Hill Climbing
    # We have a set of selected classes.
    # We can try to:
    # 1. Change residue of an existing class.
    # 2. Swap an existing class with a non-selected modulus.
    # 3. Add a non-selected modulus if it improves coverage (though greedy should have done this).
    
    non_selected_moduli = [m for m in divisors if m not in used_moduli]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try changing residues of existing classes
        for i, (m, r) in enumerate(selected_classes):
            if time.time() > deadline:
                return selected_classes
                
            best_r = r
            best_gain = -1
            m_masks = masks[m]
            
            # Current coverage without this class
            current_covered_without = covered & ~m_masks[r]
            
            for r_new in range(m):
                if r_new == r:
                    continue
                new_bits = m_masks[r_new] & ~current_covered_without
                gain = new_bits.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_r = r_new
                    if gain == N // m:
                        break
                        
            if best_gain > 0 and best_r != r:
                # Check if this actually improves total coverage
                # Total coverage with old r: covered
                # Total coverage with new r: current_covered_without | m_masks[best_r]
                old_total = covered.bit_count()
                new_total = (current_covered_without | m_masks[best_r]).bit_count()
                if new_total > old_total:
                    selected_classes[i] = [m, best_r]
                    covered = current_covered_without | m_masks[best_r]
                    improved = True
        
        # 2. Try swapping an existing class with a non-selected modulus
        # This is expensive. We can limit this to a few iterations or random sampling.
        if time.time() > deadline:
            break
            
        # Randomly pick a non-selected modulus and an existing class to swap
        if non_selected_moduli:
            for _ in range(100): # Try 100 swaps
                if time.time() > deadline:
                    break
                idx_new = random.randint(0, len(non_selected_moduli) - 1)
                m_new = non_selected_moduli[idx_new]
                
                idx_old = random.randint(0, len(selected_classes) - 1)
                m_old, r_old = selected_classes[idx_old]
                
                # Calculate coverage if we remove old and add new
                # Remove old
                covered_without_old = covered & ~masks[m_old][r_old]
                
                # Find best r for m_new given covered_without_old
                best_r_new = 0
                best_gain_new = -1
                for r_new in range(m_new):
                    new_bits = masks[m_new][r_new] & ~covered_without_old
                    gain = new_bits.bit_count()
                    if gain > best_gain_new:
                        best_gain_new = gain
                        best_r_new = r_new
                        if gain == N // m_new:
                            break
                            
                new_total = (covered_without_old | masks[m_new][best_r_new]).bit_count()
                old_total = covered.bit_count()
                
                if new_total > old_total:
                    # Perform swap
                    selected_classes[idx_old] = [m_new, best_r_new]
                    used_moduli.remove(m_old)
                    used_moduli.add(m_new)
                    non_selected_moduli.remove(m_new)
                    non_selected_moduli.append(m_old)
                    covered = covered_without_old | masks[m_new][best_r_new]
                    improved = True

    return selected_classes
