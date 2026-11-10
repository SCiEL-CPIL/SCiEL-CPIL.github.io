"""Improved program for two Sidon sets with disjoint difference sets."""
import time
import random
from itertools import combinations, combinations_with_replacement

def _is_sidon(arr):
    """Check if arr is a Sidon set (all pair sums a+b with a<=b are unique)."""
    n = len(arr)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(arr):
    """Return set of nonzero differences |b-a| for a != b in arr."""
    d = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d.add(abs(arr[j] - arr[i]))
    return d

def _greedy_construct(order, deadline, current_best_size):
    """
    Greedily build two Sidon sets with disjoint difference sets.
    order: list of candidate values to try
    Returns (A, B) or None if deadline exceeded.
    """
    sets = [[], []]
    diff_sets = [set(), set()]
    sum_sets = [set(), set()]
    
    for x in order:
        if time.time() > deadline:
            return None
        for side in (0, 1):
            if len(sets[side]) >= 64:
                continue
            # Check Sidon condition for this set
            ok = True
            for a in sets[side]:
                s = a + x
                if s in sum_sets[side]:
                    ok = False
                    break
            if ok:
                s = x + x
                if s in sum_sets[side]:
                    ok = False
            if not ok:
                continue
            # Check difference set disjointness
            new_diffs = set()
            for a in sets[side]:
                d = abs(x - a)
                if d == 0:
                    continue
                new_diffs.add(d)
                if d in diff_sets[1 - side]:
                    ok = False
                    break
            if not ok:
                continue
            # Check new_diffs don't conflict with existing diffs of same side
            # (they shouldn't since we're adding to same set, but differences
            #  within the set must also be unique? No - Sidon is about sums,
            #  differences just need to be disjoint between the two sets)
            
            # Tentatively add
            sets[side].append(x)
            for a in sets[side]:
                sum_sets[side].add(a + x)
            sum_sets[side].add(x + x)
            diff_sets[side] = diff_sets[side] | new_diffs
            
            # Check if this improves combined size
            combined = len(sets[0]) + len(sets[1])
            if combined > current_best_size[0]:
                current_best_size[0] = combined
                current_best_size[1] = (list(sets[0]), list(sets[1]))
            
            break  # Only add to one side per candidate
    
    return current_best_size[1] if current_best_size[1] else None

def _local_improve(A, B, deadline, current_best):
    """
    Try to improve by swapping elements or adding/removing.
    Returns improved (A, B) or original.
    """
    A = list(A)
    B = list(B)
    best_A, best_B = list(A), list(B)
    best_size = len(A) + len(B)
    
    # Try adding elements not in either set
    used = set(A) | set(B)
    candidates = [x for x in range(1, 257) if x not in used]
    
    for x in candidates:
        if time.time() > deadline:
            break
        for side in (0, 1):
            if len(best_A if side == 0 else best_B) >= 64:
                continue
            S = best_A if side == 0 else best_B
            O = best_B if side == 0 else best_A
            
            # Check Sidon
            ok = True
            for a in S:
                if (a + x) in {s for s in [b+c for b,c in combinations_with_replacement(S,2)]}:
                    ok = False
                    break
            if ok and (x+x) in {b+c for b,c in combinations_with_replacement(S,2)}:
                ok = False
            
            if not ok:
                continue
            
            # Check difference disjointness
            new_diffs = {abs(x - a) for a in S if x != a}
            other_diffs = {abs(b - c) for b, c in combinations(O, 2)}
            if new_diffs & other_diffs:
                continue
            
            # Try adding
            new_S = S + [x]
            new_size = len(new_S) + len(O)
            if new_size > best_size:
                # Verify fully
                if _is_sidon(new_S) and _is_sidon(O):
                    d1 = _diffs(new_S)
                    d2 = _diffs(O)
                    if not (d1 & d2):
                        if side == 0:
                            best_A = new_S
                        else:
                            best_B = new_S
                        best_size = new_size
    
    return (best_A, best_B)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_A, best_B = [], []
    best_size = 0
    
    # Multiple random restarts with different orderings
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if restart % 3 == 0:
            order = list(range(1, 257))
            random.shuffle(order)
        elif restart % 3 == 1:
            # Smaller first bias
            order = list(range(1, 257))
            random.shuffle(order)
            order.sort(key=lambda x: (x % 16, random.random()))
        else:
            # Mixed
            order = list(range(1, 257))
            random.shuffle(order)
        
        result = _greedy_construct(order, deadline, [best_size, (best_A, best_B)])
        if result is not None:
            A, B = result
            if len(A) + len(B) > best_size:
                best_A, best_B = A, B
                best_size = len(A) + len(B)
        
        # Local improvement
        if best_size > 0 and time.time() < deadline - 5:
            improved = _local_improve(best_A, best_B, deadline, [best_size, (best_A, best_B)])
            if improved is not None:
                A, B = improved
                if len(A) + len(B) > best_size:
                    best_A, best_B = A, B
                    best_size = len(A) + len(B)
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
