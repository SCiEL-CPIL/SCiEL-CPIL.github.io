"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import bisect
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    float_vals = [1.0 / d for d in denominators]
    
    def is_valid(selected):
        """Check if selected indices form a valid solution (sum strictly below 1)."""
        if not selected:
            return False
        total = Fraction(0)
        for i in selected:
            total += fracs[i]
        return total < 1
    
    def get_deficit(selected):
        """Compute the deficit 1 - sum(1/d_i) for selected indices."""
        total = Fraction(0)
        for i in selected:
            total += fracs[i]
        return 1 - total

    # --- MEET-IN-THE-MIDDLE APPROACH ---
    # LCM of denominators 2..48
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)
        
    A = [L // d for d in denominators]
    
    # Split into two halves
    n1 = n // 2
    n2 = n - n1
    
    # Generate subset sums for first half with masks
    # We only care about sums strictly less than L
    pairs1 = [(0, 0)]
    for i in range(n1):
        if time.time() > deadline:
            break
        new_pairs = []
        ai = A[i]
        for s, m in pairs1:
            new_pairs.append((s, m))
            ns = s + ai
            if ns < L:
                new_pairs.append((ns, m | (1 << i)))
        pairs1 = new_pairs
    
    # Generate subset sums for second half with masks
    pairs2 = [(0, 0)]
    for i in range(n1, n):
        if time.time() > deadline:
            break
        new_pairs = []
        ai = A[i]
        for s, m in pairs2:
            new_pairs.append((s, m))
            ns = s + ai
            if ns < L:
                new_pairs.append((ns, m | (1 << i)))
        pairs2 = new_pairs
    
    # Sort pairs2 by sum
    pairs2.sort(key=lambda x: x[0])
    sums2_sorted = [p[0] for p in pairs2]
    
    best_sum = 0
    best_mask = 0
    
    max_s2 = sums2_sorted[-1] if sums2_sorted else 0
    
    # For each sum in first half, find best compatible sum in second half
    for s1, m1 in pairs1:
        if time.time() > deadline:
            break
        # Pruning: if even max s2 can't beat current best, skip
        if s1 + max_s2 <= best_sum:
            continue
            
        limit = L - s1
        idx = bisect.bisect_left(sums2_sorted, limit)
        if idx > 0:
            s2, m2 = pairs2[idx - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_mask = m1 | m2
    
    # Extract indices from best_mask
    if best_mask != 0:
        best_selected = set()
        for i in range(n):
            if best_mask & (1 << i):
                best_selected.add(i)
        
        # Verify with Fraction
        if is_valid(best_selected):
            best_deficit = get_deficit(best_selected)
        else:
            # Should not happen, but fallback
            best_selected = {0}
            best_deficit = Fraction(1, 2)
    else:
        best_selected = {0}
        best_deficit = Fraction(1, 2)
    
    # --- LOCAL SEARCH REFINEMENT ---
    # Use the MITM result as starting point, then try to improve
    selected = set(best_selected)
    total_float = sum(float_vals[i] for i in selected)
    total_frac = sum(fracs[i] for i in selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding single elements
        for i in range(n):
            if time.time() > deadline:
                break
            if i not in selected:
                new_float = total_float + float_vals[i]
                if new_float < 0.999999999 and new_float > total_float:
                    new_selected = selected | {i}
                    if is_valid(new_selected):
                        new_deficit = get_deficit(new_selected)
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            best_selected = new_selected
                            selected = new_selected
                            total_float = new_float
                            improved = True
                            break
        
        if time.time() > deadline:
            break
        
        # Try removing single elements (to make room for better additions)
        for rem in list(selected):
            if time.time() > deadline:
                break
            new_selected = selected - {rem}
            new_float = total_float - float_vals[rem]
            # After removal, try adding elements
            for add in range(n):
                if time.time() > deadline:
                    break
                if add not in new_selected:
                    new_float2 = new_float + float_vals[add]
                    if new_float2 < 0.999999999 and new_float2 > total_float:
                        new_selected2 = new_selected | {add}
                        if is_valid(new_selected2):
                            new_deficit = get_deficit(new_selected2)
                            if new_deficit < best_deficit:
                                best_deficit = new_deficit
                                best_selected = new_selected2
                                selected = new_selected2
                                total_float = new_float2
                                improved = True
                                break
                if time.time() > deadline:
                    break
            if improved:
                break
        
        if time.time() > deadline:
            break
    
    # Final verification
    if not is_valid(best_selected):
        # Fallback: use a simple valid solution
        best_selected = {0}
    
    return sorted(best_selected)
# EVOLVE-BLOCK-END
