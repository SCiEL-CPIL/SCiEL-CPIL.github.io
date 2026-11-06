def construct():
    import time
    from math import gcd

    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s

    def elapsed():
        return time.time() - start_time

    def time_left():
        return DEADLINE - elapsed()

    # Precompute gcd table for [1,128]
    n = 128
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        """Check if triple (a,b,c) violates: all three pairwise gcds equal."""
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]

    def can_add(S, x):
        """Check if x can be added to set S without creating a bad triple."""
        for a in S:
            for b in S:
                if a < b and is_bad_triple(a, b, x):
                    return False
        return True

    def count_bad_triples_per_element(selected):
        """Count how many bad triples each element participates in."""
        sel = list(selected)
        m = len(sel)
        counts = [0] * m
        for i in range(m):
            for j in range(i + 1, m):
                d_ij = gcd_table[sel[i]][sel[j]]
                for k in range(j + 1, m):
                    d_ik = gcd_table[sel[i]][sel[k]]
                    if d_ik != d_ij:
                        continue
                    d_jk = gcd_table[sel[j]][sel[k]]
                    if d_jk == d_ij:
                        counts[i] += 1
                        counts[j] += 1
                        counts[k] += 1
        return counts

    # Strategy 1: Greedy removal from full set
    # Start with all 128 elements, repeatedly remove the element in the most bad triples
    selected = list(range(1, n + 1))
    
    while time_left() > 5.0:
        if len(selected) < 2:
            break
        counts = count_bad_triples_per_element(selected)
        max_count = max(counts)
        if max_count == 0:
            break
        # Remove the element with highest bad-triple count; tie-break: prefer removing smaller values
        max_indices = [i for i, c in enumerate(counts) if c == max_count]
        # Among tied, remove the smallest value (keep larger numbers)
        remove_idx = min(max_indices, key=lambda i: selected[i])
        selected.pop(remove_idx)
    
    best_set = set(selected)
    best_size = len(best_set)

    # Strategy 2: Local search with single-element swaps
    # Try to swap out an element for one that allows adding more
    def local_search(S):
        S = list(S)
        improved = True
        while improved and time_left() > 3.0:
            improved = False
            for i in range(len(S)):
                if time_left() < 2.0:
                    break
                removed = S[i]
                base = [x for x in S if x != removed]
                # Try adding any element from 1..128 not in base
                for x in range(1, n + 1):
                    if x in base:
                        continue
                    # Check if x can be added to base
                    valid = True
                    for a in base:
                        for b in base:
                            if a < b and is_bad_triple(a, b, x):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # New set: base + x, size is same as S
                        # But maybe this enables further additions?
                        # For now, accept if it doesn't decrease size (same size)
                        # Actually, we want to try to INCREASE size, so this swap alone doesn't help
                        # unless it enables adding more. Let's try a two-step: swap then add.
                        # Simpler: just try to add x to S directly (no removal)
                        pass
            # Try adding elements without removing
            for x in range(1, n + 1):
                if time_left() < 1.0:
                    break
                if x not in S:
                    valid = True
                    for a in S:
                        for b in S:
                            if a < b and is_bad_triple(a, b, x):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        S.append(x)
                        improved = True
                        break
        return S

    best_set = local_search(list(best_set))
    best_size = len(best_set)

    # Strategy 3: Try multiple greedy removals with different tie-breaking
    # and compare results
    for tie_break in ['smallest', 'largest', 'random']:
        if time_left() < 10.0:
            break
        
        selected2 = list(range(1, n + 1))
        while time_left() > 5.0:
            if len(selected2) < 2:
                break
            counts = count_bad_triples_per_element(selected2)
            max_count = max(counts)
            if max_count == 0:
                break
            max_indices = [i for i, c in enumerate(counts) if c == max_count]
            if tie_break == 'smallest':
                remove_idx = min(max_indices, key=lambda i: selected2[i])
            elif tie_break == 'largest':
                remove_idx = max(max_indices, key=lambda i: selected2[i])
            else:
                import random
                remove_idx = random.choice(max_indices)
            selected2.pop(remove_idx)
        
        # Local search on this result
        selected2 = local_search(selected2)
        if len(selected2) > best_size:
            best_size = len(selected2)
            best_set = set(selected2)

    # Final polish: try to add any remaining elements
    for x in range(1, n + 1):
        if time_left() < 1.0:
            break
        if x not in best_set:
            valid = True
            for a in best_set:
                for b in best_set:
                    if a < b and is_bad_triple(a, b, x):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_set.add(x)

    return list(best_set)
