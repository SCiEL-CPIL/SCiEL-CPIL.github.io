def construct():
    import time
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s

    def elapsed_ok():
        return (time.time() - t0) < DEADLINE

    from fractions import Fraction

    # Denominators: 2, 3, ..., 48 -> indices 0..46
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Precompute suffix sums: suffix_sum[i] = sum of 1/denoms[j] for j from i to n-1
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / denoms[i]

    best_deficit = None
    best_indices = None

    def consider(indices, current_sum_frac):
        nonlocal best_deficit, best_indices
        if not indices:
            return
        s = current_sum_frac
        if s < 1:
            d = 1 - s
            if best_deficit is None or d < best_deficit:
                best_deficit = d
                best_indices = list(indices)

    # Fallback greedy
    def greedy():
        sel, tot = [], Fraction(0)
        for i in range(n):
            d = denoms[i]
            if tot + Fraction(1, d) < 1:
                sel.append(i)
                tot += Fraction(1, d)
        return sel

    # Branch and bound: try including/excluding each item
    # Prune if: current_sum + suffix_sum[remaining] < best_sum (can't improve)
    # Or: current_sum >= 1 (invalid, must be strictly below 1)
    
    def bbs(idx, selected, current_sum):
        nonlocal best_deficit, best_indices
        if not elapsed_ok():
            return
        
        # Base case: all items considered
        if idx == n:
            if selected and current_sum < 1:
                d = 1 - current_sum
                if best_deficit is None or d < best_deficit:
                    best_deficit = d
                    best_indices = list(selected)
            return
        
        # Pruning: even if we take all remaining, can't beat current best
        if best_deficit is not None:
            max_possible = current_sum + suffix_sum[idx]
            if max_possible < 1 - best_deficit:
                return  # Can't improve
        
        # Option 1: Exclude item idx
        bbs(idx + 1, selected, current_sum)
        
        # Option 2: Include item idx (if it keeps sum < 1)
        new_sum = current_sum + Fraction(1, denoms[idx])
        if new_sum < 1:
            selected.append(idx)
            bbs(idx + 1, selected, new_sum)
            selected.pop()

    # Start with greedy to get an initial good bound
    sel, tot = [], Fraction(0)
    for i in range(n):
        d = denoms[i]
        if tot + Fraction(1, d) < 1:
            sel.append(i)
            tot += Fraction(1, d)
    if sel and tot < 1:
        best_deficit = 1 - tot
        best_indices = list(sel)

    # Run B&B
    bbs(0, [], Fraction(0))

    if best_indices is None:
        best_indices = greedy()
    return best_indices
