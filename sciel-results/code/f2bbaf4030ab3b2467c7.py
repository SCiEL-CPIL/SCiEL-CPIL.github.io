<<<<<<< SEARCH
import time
from fractions import Fraction
from math import gcd
import bisect

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s wall-clock limit

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Split into two halves
    mid = n // 2  # 23
    n1 = mid
    n2 = n - mid  # 24

    # Floating point weights for fast search
    w_float = [1.0 / d for d in denominators]
    w_left = w_float[:n1]
    w_right = w_float[n1:]

    # Generate all subset sums (float) for each half
    def gen_subsets_float(weights):
        """Generate all subset sums as floats. Returns list of (sum, mask)."""
        results = [(0.0, 0)]  # empty set
        for i, wt in enumerate(weights):
            bit = 1 << i
            # For each existing subset, try adding weight i
            new_entries = []
            for s, m in results:
                ns = s + wt
                new_entries.append((ns, m | bit))
            results.extend(new_entries)
        return results

    # Generate for left half
    left_subsets = gen_subsets_float(w_left)
    
    if time.time() > deadline - 10:
        # Fallback: simple greedy
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Generate for right half
    right_subsets = gen_subsets_float(w_right)
    
    if time.time() > deadline - 10:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Sort right subsets by sum for binary search
    right_sorted = sorted(right_subsets, key=lambda x: x[0])
    right_sums = [x[0] for x in right_sorted]
    right_masks = [x[1] for x in right_sorted]

    # For each left subset, find the best right subset such that total < 1
    best_total = -1.0
    best_left_mask = 0
    best_right_mask = 0

    for ls, lm in left_subsets:
        if time.time() > deadline - 5:
            break
        
        # Need rs < 1 - ls
        bound = 1.0 - ls
        if bound <= 0:
            continue
        # Find rightmost index where right_sums[idx] < bound
        # bisect_left gives first index where right_sums[idx] >= bound
        idx = bisect.bisect_left(right_sums, bound) - 1
        if idx < 0:
            continue
        
        rs = right_sums[idx]
        rm = right_masks[idx]
        total = ls + rs
        
        # Total must be strictly less than 1 (guaranteed by construction)
        # and non-empty overall
        if total > best_total and (lm | rm) != 0:
            best_total = total
            best_left_mask = lm
            best_right_mask = rm

    # Also consider: maybe a single half alone is better
    # Left alone
    for ls, lm in left_subsets:
        if lm != 0 and ls > best_total:
            best_total = ls
            best_left_mask = lm
            best_right_mask = 0
    # Right alone
    for rs, rm in right_subsets:
        if rm != 0 and rs > best_total:
            best_total = rs
            best_left_mask = 0
            best_right_mask = rm

    # Reconstruct selected indices
    selected = []
    for i in range(n1):
        if best_left_mask & (1 << i):
            selected.append(i)
    for i in range(n2):
        if best_right_mask & (1 << i):
            selected.append(n1 + i)

    if not selected:
        # Fallback greedy
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Exact verification with Fraction
    total_frac = Fraction(0)
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])
    
    if total_frac >= 1:
        # Should not happen, but fallback
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    return selected
=======
import time
from fractions import Fraction
from math import gcd
import bisect

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s wall-clock limit

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Fallback greedy function
    def greedy_fallback():
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Strategy: Use exact rational meet-in-the-middle
    # Split into two halves, generate all subset sums as Fractions
    # For the smaller half, do exhaustive enumeration
    # For the larger half, use binary search on sorted sums
    
    # Try different splits to find one that fits in time
    # Start with a 20/27 split (smaller first half for exhaustive search)
    
    best_result = None
    best_deficit = None  # We want to minimize deficit = 1 - sum, i.e., maximize sum
    
    for split in [20, 21, 22, 23, 24, 25]:
        n1 = split
        n2 = n - n1
        
        if n1 > n2:
            # Swap to keep n1 smaller
            n1, n2 = n2, n1
        
        # Generate all subset sums for the smaller half (exact)
        # This is 2^n1 subsets
        if n1 > 24:  # Too many to enumerate exactly
            continue
        
        # Generate subset sums for half 1 (indices 0..n1-1)
        left_sums = [(Fraction(0), 0)]  # (sum, mask)
        for i in range(n1):
            bit = 1 << i
            frac_i = Fraction(1, denominators[i])
            new_entries = []
            for s, m in left_sums:
                new_entries.append((s + frac_i, m | bit))
            left_sums.extend(new_entries)
        
        if time.time() > deadline - 20:
            break
        
        # Generate subset sums for half 2 (indices n1..n-1)
        right_sums = [(Fraction(0), 0)]
        for i in range(n2):
            bit = 1 << i
            frac_i = Fraction(1, denominators[n1 + i])
            new_entries = []
            for s, m in right_sums:
                new_entries.append((s + frac_i, m | bit))
            right_sums.extend(new_entries)
        
        if time.time() > deadline - 10:
            break
        
        # Sort right sums by value for binary search
        right_sorted = sorted(right_sums, key=lambda x: x[0])
        right_vals = [x[0] for x in right_sorted]
        
        # For each left subset, find best right subset
        # We need left_sum + right_sum < 1, maximize total
        for ls, lm in left_sums:
            if time.time() > deadline - 5:
                break
            
            # Need rs < 1 - ls
            bound = Fraction(1) - ls
            if bound <= 0:
                continue
            
            # Find rightmost index where right_vals[idx] < bound
            idx = bisect.bisect_left(right_vals, bound) - 1
            if idx < 0:
                continue
            
            rs, rm = right_sorted[idx]
            total = ls + rs
            
            # Must be non-empty overall
            if (lm | rm) == 0:
                continue
            
            # Check if this is better (closer to 1)
            deficit = Fraction(1) - total
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_result = (lm, rm, n1)
        
        if time.time() > deadline - 10:
            break
        
        # Also check single-half solutions
        for ls, lm in left_sums:
            if lm != 0:
                deficit = Fraction(1) - ls
                if best_deficit is None or deficit < best_deficit:
                    best_deficit = deficit
                    best_result = (lm, 0, n1)
        
        for rs, rm in right_sums:
            if rm != 0:
                deficit = Fraction(1) - rs
                if best_deficit is None or deficit < best_deficit:
                    best_deficit = deficit
                    best_result = (0, rm, n1)
        
        if time.time() > deadline - 10:
            break
        
        # If we found a very good result, we can stop
        if best_deficit is not None and best_deficit.numerator <= 1:
            break
    
    # Reconstruct from best result
    if best_result is not None:
        lm, rm, n1 = best_result
        selected = []
        for i in range(n1):
            if lm & (1 << i):
                selected.append(i)
        for i in range(n - n1):
            if rm & (1 << i):
                selected.append(n1 + i)
        
        if selected:
            # Verify
            total_frac = Fraction(0)
            for idx in selected:
                total_frac += Fraction(1, denominators[idx])
            if total_frac < 1:
                return selected
    
    # Fallback
    return greedy_fallback()
>>>>>>> REPLACE
