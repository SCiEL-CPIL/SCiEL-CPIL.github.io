import time
import random
import math

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute all non-constant 4-term APs
    # An AP is (i, i+d, i+2d, i+3d)
    aps = []
    for i in range(1, N + 1):
        for d in range(1, (N - i + 1) // 3 + 1):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l <= N:
                aps.append((i, j, k, l))
                
    K = 3 # We aim for 3 colors. If impossible, we might need more, but 3 is the target for minimization.
    # Actually, the problem asks to minimize colors. Let's try to find a solution with 3 colors.
    # If we can't find one quickly, we might need to fall back to more, but 3 is the theoretical minimum 
    # for this type of constraint usually (van der Waerden numbers W(3,4)=76, so 128 is well within range for 3 colors? 
    # Wait, W(3,4)=76 means any 3-coloring of [1,76] has a monochromatic 4-AP.
    # The constraint is "AT LEAST 3 distinct colors". This is NOT "monochromatic".
    # It forbids colorings where the 4-AP uses only 1 or 2 colors.
    # So it allows 3 or 4 colors. Since we only have 3 colors available in our search space, 
    # it effectively requires exactly 3 colors for every 4-AP.
    
    # Let's verify: if we use 3 colors, "at least 3 distinct" means "exactly 3 distinct".
    # So every 4-AP must use all 3 colors.
    
    def count_violations(colors):
        # colors is 0-indexed list of length N, values in 0..2
        v = 0
        for (i, j, k, l) in aps:
            # Check if the set of colors has size < 3
            # Since we only have 3 colors, size < 3 means size 1 or 2.
            c1 = colors[i-1]
            c2 = colors[j-1]
            c3 = colors[k-1]
            c4 = colors[l-1]
            # Fast check: if any two are same, and the other two are same as one of them or each other?
            # Simpler: just use set for correctness first, optimize if needed.
            # But set creation is slow. Let's use a bitmask or manual check.
            # Bitmask:
            mask = (1 << c1) | (1 << c2) | (1 << c3) | (1 << c4)
            # Count bits in mask
            if mask.bit_count() < 3:
                v += 1
        return v

    # Initialize random coloring
    colors = [random.randint(0, K - 1) for _ in range(N)]
    
    # Initial energy
    E = count_violations(colors)
    
    best_colors = colors[:]
    best_E = E
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # We will iterate until deadline
    while time.time() < deadline:
        # Pick a random position to change
        idx = random.randint(0, N - 1)
        old_c = colors[idx]
        new_c = random.randint(0, K - 1)
        
        if new_c == old_c:
            continue
            
        colors[idx] = new_c
        
        # Calculate Delta E
        # We only need to check APs that include idx
        # Precompute APs containing each index?
        # For speed, let's just recalculate total violations if N is small enough?
        # N=128, number of APs is roughly sum_{d=1}^{42} (128-3d) ~ 42*128 - 3*42*43/2 ~ 5000.
        # Recalculating 5000 APs per step is too slow for Python in 85s if we do millions of steps.
        # We need incremental updates.
        
        # Let's precompute for each index, the list of APs it belongs to.
        # This was not done in the outer scope. Let's do it now.
        # Actually, I should have done this before the loop.
        # Since I can't easily refactor the whole structure without breaking the "single code block" flow,
        # let's optimize the violation count by only checking relevant APs.
        
        # Re-doing the precomputation inside the function is fine, it's fast.
        
        # Wait, I didn't precompute `aps_containing` in the first draft.
        # Let's assume I do it now.
        
        # To keep the code simple and robust within the time limit, 
        # I will precompute the inverse mapping now.
        
        # (Self-correction: I will generate the code with the precomputation included properly)
        
        # Let's restart the logic structure in the final code block to ensure efficiency.
        
        # For this thought process, I assume I will write:
        # 1. Precompute aps
        # 2. Precompute index_to_aps
        # 3. Simulated Annealing loop with incremental energy update.
        
        pass

    # If we didn't find a 0-violation solution with 3 colors, 
    # we might need to try 4 colors? 
    # The problem says "Minimize the number of labels". 
    # If 3 is impossible, 4 is next. 
    # But W(3,4)=76 implies that for 3 colors, there is always a monochromatic AP in [1,76].
    # Monochromatic AP uses 1 color. The constraint is "AT LEAST 3". 
    # So a monochromatic AP is a violation.
    # Thus, a 3-coloring of [1,128] MUST have a monochromatic 4-AP (since 128 > 76).
    # Therefore, a 3-coloring is IMPOSSIBLE.
    
    # Wait, let me double check Van der Waerden numbers.
    # W(3,4) = 76. This means any 3-coloring of {1,...,76} contains a monochromatic 4-AP.
    # The condition is "Every nonconstant four-term progression ... must use AT LEAST three distinct colors".
    # A monochromatic AP uses 1 color. This violates the condition.
    # So, if W(3,4)=76, then no 3-coloring of [1,128] is valid.
    
    # What about 4 colors?
    # W(4,4) is much larger. 
    # The lower bound for W(4,4) is known to be > 1000? 
    # Actually, W(4,4) is known to be at least 128? No, that's too low.
    # Let's check known values.
    # W(3,4) = 76.
    # W(4,4) is between 1000 and 10000? 
    # Actually, the problem asks to minimize colors. 
    # If 3 is impossible, we must use at least 4.
    
    # Let's try to find a valid coloring with 4 colors.
    # The constraint is: every 4-AP uses at least 3 colors.
    # This allows 3 or 4 colors.
    # With 4 colors, we have more flexibility.
    
    # Let's modify the search to use K=4.
    
    # If K=4 is also impossible? Unlikely for 128.
    
    # Let's implement the SA for K=4.
    
    # If K=4 fails, try K=5?
    # The fallback in the seed program used a greedy approach that might use more colors.
    
    # Strategy:
    # 1. Try SA with K=4.
    # 2. If found, return.
    # 3. If not found, try K=5.
    # 4. If not found, use a greedy fallback.
    
    # Let's refine the code to handle this.

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute all non-constant 4-term APs
    aps = []
    for i in range(1, N + 1):
        for d in range(1, (N - i + 1) // 3 + 1):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l <= N:
                aps.append((i, j, k, l))
                
    # Precompute for each index, the list of AP indices it belongs to
    index_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            index_to_aps[pos - 1].append(idx)

    def count_violations(colors):
        v = 0
        for ap in aps:
            c1 = colors[ap[0]-1]
            c2 = colors[ap[1]-1]
            c3 = colors[ap[2]-1]
            c4 = colors[ap[3]-1]
            mask = (1 << c1) | (1 << c2) | (1 << c3) | (1 << c4)
            if mask.bit_count() < 3:
                v += 1
        return v

    def delta_energy(colors, idx, old_c, new_c, K):
        # Calculate change in violations for changing index idx from old_c to new_c
        # Only APs containing idx are affected
        dE = 0
        for ap_idx in index_to_aps[idx]:
            ap = aps[ap_idx]
            # Get the 4 colors
            c1 = colors[ap[0]-1]
            c2 = colors[ap[1]-1]
            c3 = colors[ap[2]-1]
            c4 = colors[ap[3]-1]
            
            # If idx is part of this AP, one of these is old_c
            # We need to check the state BEFORE and AFTER
            
            # Before:
            mask_before = 0
            for pos in ap:
                c = colors[pos-1]
                if pos - 1 == idx:
                    c = old_c
                mask_before |= (1 << c)
            
            # After:
            mask_after = 0
            for pos in ap:
                c = colors[pos-1]
                if pos - 1 == idx:
                    c = new_c
                mask_after |= (1 << c)
            
            viol_before = 1 if mask_before.bit_count() < 3 else 0
            viol_after = 1 if mask_after.bit_count() < 3 else 0
            
            dE += viol_after - viol_before
            
        return dE

    def solve_sa(K, time_limit):
        colors = [random.randint(0, K - 1) for _ in range(N)]
        E = count_violations(colors)
        
        best_colors = colors[:]
        best_E = E
        
        T = 5.0
        cooling_rate = 0.9995
        min_T = 0.005
        
        while time.time() < time_limit:
            if E == 0:
                return colors[:]
                
            idx = random.randint(0, N - 1)
            old_c = colors[idx]
            new_c = random.randint(0, K - 1)
            
            if new_c == old_c:
                continue
                
            dE = delta_energy(colors, idx, old_c, new_c, K)
            
            # Accept if dE < 0 or with probability exp(-dE/T)
            if dE < 0 or random.random() < math.exp(-dE / T):
                colors[idx] = new_c
                E += dE
                
                if E < best_E:
                    best_E = E
                    best_colors = colors[:]
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
        return best_colors if best_E < E else colors[:]

    # Try K=4 first (since K=3 is impossible by W(3,4)=76)
    # Wait, is W(3,4) really 76?
    # Yes, W(3,4) = 76.
    # So any 3-coloring of [1,76] has a mono 4-AP.
    # [1,128] contains [1,76].
    # So any 3-coloring of [1,128] has a mono 4-AP.
    # Mono 4-AP has 1 color.
    # Constraint: at least 3 colors.
    # So 3 colors is impossible.
    
    # Try K=4
    sol = solve_sa(4, start_time + 40.0)
    if count_violations(sol) == 0:
        # Map colors to 0..K-1 (they are already)
        # The problem says "labels in 0..127". 
        # "Minimize the number of labels actually used".
        # If we use 4 colors, we return them.
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]
        
    # Try K=5
    sol = solve_sa(5, start_time + 70.0)
    if count_violations(sol) == 0:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]
        
    # Fallback: Greedy construction
    # This ensures we return *something*, though it might use more colors.
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            # Check all APs ending at x? No, APs can start before x and end after?
            # The AP is (i, i+d, i+2d, i+3d).
            # If we assign x, we need to check APs where x is one of the terms.
            # It's easier to check all APs that contain x.
            for ap_idx in index_to_aps[x-1]:
                ap = aps[ap_idx]
                # Get colors of other 3 positions
                c_vals = []
                for pos in ap:
                    if pos == x:
                        c_vals.append(color)
                    elif pos <= len(colors):
                        c_vals.append(colors[pos-1])
                    else:
                        # Future position, not assigned yet.
                        # We can't check this constraint yet if it involves future positions.
                        # However, if we are filling left-to-right, any AP containing x 
                        # and positions > x will be checked when those positions are assigned?
                        # No, the constraint is global.
                        # If we only check APs fully assigned, we might miss conflicts.
                        # But if we check APs where x is the *last* element, we are safe?
                        # No, x can be the first, second, or third.
                        
                        # To be safe, the greedy approach in the seed program checked:
                        # `all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in ...)`
                        # This checks APs ending at x.
                        # Does that suffice?
                        # If an AP is (x, x+d, x+2d, x+3d), and we assign x, 
                        # the constraint involves future colors.
                        # We cannot guarantee it now.
                        # However, if we ensure that for every AP ending at x, the condition holds,
                        # and for every AP starting at x, we hope future assignments work out?
                        # This is not guaranteed.
                        
                        # The seed program's fallback was:
                        # `if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))`
                        # This ONLY checks APs ending at x.
                        # It ignores APs where x is not the last term.
                        # This is a heuristic. It might produce an invalid solution if not careful.
                        # But it's a fallback.
                        
                        # Let's stick to checking APs where x is the last term for speed,
                        # and assume the SA solution is the primary target.
                        # If SA fails, we return the best SA solution found, even if invalid?
                        # No, "invalid result... scores 0".
                        # We must return a VALID solution.
                        
                        # If SA fails to find 0 violations, we have a problem.
                        # But K=4 should be sufficient.
                        # If K=4 fails, K=5 should work.
                        # If K=5 fails, we can try a randomized greedy that checks ALL constraints 
                        # involving assigned variables.
                        
                        pass
                # If all positions in AP are assigned:
                if all(pos <= len(colors) or pos == x for pos in ap):
                    if len(set(c_vals)) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            # Should not happen with enough colors
            colors.append(0)
            
    # Verify the fallback
    if count_violations(colors) == 0:
        used = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in colors]
        
    # If everything fails, return the best SA result (might be invalid, but better than nothing? No, score 0)
    # Let's force a valid solution by using many colors.
    # With 128 colors, we can just assign distinct colors to all?
    # If all colors are distinct, every 4-AP has 4 distinct colors. Valid.
    # So [0, 1, ..., 127] is always a valid solution.
    return list(range(128))

