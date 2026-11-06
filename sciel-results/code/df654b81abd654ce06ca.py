import numpy as np
import time

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate divisors
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # Precompute coverage masks for each modulus and residue?
    # That would be sum(m for m in divisors) * m entries? No.
    # For each m, there are m residues.
    # Total entries = sum(m for m in divisors).
    # sum(m for m in divisors) is sigma(51975) approx 120k.
    # We can store for each (m, r) a boolean array of length base?
    # That would be 120k * 52k bytes = 6 GB. Too much.
    
    # Instead, we compute coverage on the fly using slicing.
    # covered = np.zeros(base, dtype=bool)
    
    covered = np.zeros(base, dtype=bool)
    classes = []
    
    # 1. Greedy Initialization
    # We maintain a list of available moduli
    available_moduli = divisors.copy()
    
    while available_moduli:
        if time.time() > deadline:
            break
            
        best_gain = 0
        best_m = None
        best_r = None
        
        # To speed up, we can sample if list is large, but 47 is small.
        for m in available_moduli:
            # Find best r for this m
            # Count uncovered residues for each residue class
            # We can do this by iterating r and counting covered[r::m]
            # This is O(m * base/m) = O(base) per m.
            # Total O(base * num_moduli) = 52k * 47 ~ 2.5M ops per step.
            # Steps ~ 47. Total ~ 100M ops. Might be slow in Python loop.
            
            # Optimization: Use numpy
            # uncovered_mask = ~covered
            # For a given m, we want to find r that maximizes sum(uncovered_mask[r::m])
            # We can reshape uncovered_mask? No, it's strided.
            
            # Let's try a faster way:
            # For each m, we can compute the counts for all r at once?
            # Not easily with numpy for arbitrary strides without complex indexing.
            
            # Let's stick to the loop but optimize the inner count.
            # count = 0
            # for r in range(m):
            #    c = np.count_nonzero(~covered[r::m])
            
            # Actually, the seed program used bytearray which is faster than numpy for small counts?
            # Let's use numpy boolean array.
            
            max_count = 0
            best_r_m = 0
            # We can break early if we find a perfect cover? Unlikely.
            for r in range(m):
                # Count uncovered in this class
                # Using slicing
                c = np.count_nonzero(~covered[r::m])
                if c > max_count:
                    max_count = c
                    best_r_m = r
            
            if max_count > best_gain:
                best_gain = max_count
                best_m = m
                best_r = best_r_m
        
        if best_m is None or best_gain == 0:
            break
            
        # Add the class
        classes.append([best_m, best_r])
        covered[best_r::best_m] = True
        available_moduli.remove(best_m)
        
        # Check if all covered
        if np.all(covered):
            break

    # 2. Local Search (Swapping)
    # State: classes list, covered array
    # We try to swap a selected modulus with an unselected one.
    
    selected_moduli = {c[0] for c in classes}
    unselected_moduli = [m for m in divisors if m not in selected_moduli]
    
    # Helper to compute best residue for a modulus given current covered
    def find_best_residue(m, covered_arr):
        max_count = 0
        best_r = 0
        for r in range(m):
            c = np.count_nonzero(~covered_arr[r::m])
            if c > max_count:
                max_count = c
                best_r = r
        return best_r, max_count

    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_uncovered = np.count_nonzero(~covered)
        
        # Try adding unselected moduli
        for m in unselected_moduli[:]: # Copy list
            if time.time() > deadline:
                break
            r, gain = find_best_residue(m, covered)
            if gain > 0:
                # Add it
                classes.append([m, r])
                covered[r::m] = True
                unselected_moduli.remove(m)
                selected_moduli.add(m)
                improved = True
                current_uncovered = np.count_nonzero(~covered)
                if current_uncovered == 0:
                    return classes

        # Try swapping
        # For each selected m_out, try swapping with each unselected m_in
        # To avoid O(N^2) complexity with high constant, we can be selective.
        # But N=47 is small. 47*47 = 2209 swaps to check.
        
        for i, m_out in enumerate(selected_moduli):
            if time.time() > deadline:
                break
            
            # Temporarily remove m_out
            r_out = None
            for c in classes:
                if c[0] == m_out:
                    r_out = c[1]
                    break
            
            # Save state
            # We need to revert if swap is not beneficial
            # Easier: compute new uncovered count virtually?
            # Or just do it and revert.
            
            covered[r_out::m_out] = False
            new_uncovered = np.count_nonzero(~covered)
            
            best_swap_gain = 0
            best_m_in = None
            best_r_in = None
            
            for m_in in unselected_moduli:
                r_in, gain_in = find_best_residue(m_in, covered)
                # Gain is number of newly covered residues
                # Net change = gain_in (covered) - (new_uncovered - current_uncovered) (uncovered added back)
                # Actually, we removed m_out, so we uncovered some residues.
                # Let lost = current_uncovered - (count before removing m_out)
                # Wait, we have `current_uncovered` from before removal?
                # Let's trace:
                # U_old: count of uncovered before removal.
                # Remove m_out: U_temp = U_old + lost
                # Add m_in: U_new = U_temp - gained
                # We want U_new < U_old => gained > lost
                
                lost = new_uncovered - current_uncovered # This is the number of residues that were ONLY covered by m_out? No.
                # If a residue was covered by m_out AND another class, removing m_out doesn't uncover it.
                # So `lost` is the number of residues that became uncovered.
                
                # We want to find m_in that covers the most of these `lost` residues?
                # `find_best_residue` finds the residue class that covers the most *currently uncovered* residues.
                # The currently uncovered residues are exactly the `lost` ones (plus those already uncovered).
                # So maximizing `gain_in` is correct.
                
                if gain_in > best_swap_gain:
                    best_swap_gain = gain_in
                    best_m_in = m_in
                    best_r_in = r_in
            
            # Check if swap is beneficial
            if best_m_in is not None and best_swap_gain > lost:
                # Perform swap
                # Update classes list
                for c in classes:
                    if c[0] == m_out:
                        c[1] = best_r_in # Wait, we need to change modulus too
                        # Actually, we should replace the entry
                        c[0] = best_m_in
                        c[1] = best_r_in
                        break
                
                # Update covered
                covered[best_r_in::best_m_in] = True
                
                # Update sets
                selected_moduli.remove(m_out)
                selected_moduli.add(best_m_in)
                unselected_moduli.remove(best_m_in)
                unselected_moduli.append(m_out)
                
                current_uncovered = np.count_nonzero(~covered)
                improved = True
                if current_uncovered == 0:
                    return classes
            else:
                # Revert
                covered[r_out::m_out] = True

    return classes
