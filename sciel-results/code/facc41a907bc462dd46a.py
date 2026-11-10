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

def _can_add(sets, diffs, sums, x, side):
    """Check if x can be added to sets[side] without violating constraints."""
    # Check Sidon condition
    for a in sets[side]:
        s = a + x
        if s in sums[side]:
            return False
    if x + x in sums[side]:
        return False
    
    # Check difference disjointness
    for a in sets[side]:
        d = abs(x - a)
        if d in diffs[1 - side]:
            return False
    return True

def _greedy_construct(order, deadline):
    """Greedy construction given an order of candidates."""
    sets = [[], []]
    diffs = [set(), set()]
    sums = [set(), set()]
    
    for x in order:
        if time.time() > deadline:
            break
        for side in (0, 1):
            if _can_add(sets, diffs, sums, x, side):
                sets[side].append(x)
                for a in sets[side]:
                    sums[side].add(a + x)
                sums[side].add(x + x)
                for a in sets[side]:
                    if a != x:
                        diffs[side].add(abs(x - a))
    
    return sets

def _greedy_cost_based(available, deadline):
    """Greedy construction that picks the element with minimum conflict cost at each step."""
    sets = [[], []]
    diffs = [set(), set()]
    sums = [set(), set()]
    remaining = list(available)
    
    while remaining:
        if time.time() > deadline:
            break
        
        best_x = None
        best_side = None
        best_cost = float('inf')
        
        for x in remaining:
            for side in (0, 1):
                if _can_add(sets, diffs, sums, x, side):
                    # Count how many other elements this x conflicts with (for future)
                    cost = 0
                    for a in sets[side]:
                        d = abs(x - a)
                        if d in diffs[1 - side]:
                            cost += 1
                    if cost < best_cost or (cost == best_cost and (best_x is None or x < best_x)):
                        best_cost = cost
                        best_x = x
                        best_side = side
        
        if best_x is None:
            break
        
        remaining.remove(best_x)
        sets[best_side].append(best_x)
        for a in sets[best_side]:
            sums[best_side].add(a + best_x)
        sums[best_side].add(best_x + best_x)
        for a in sets[best_side]:
            if a != best_x:
                diffs[best_side].add(abs(best_x - a))
    
    return sets

def _local_improve(sets, deadline):
    """Try to add more elements or swap elements to improve combined size."""
    best = [list(sets[0]), list(sets[1])]
    best_score = len(best[0]) + len(best[1])
    
    for iteration in range(100):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding an element
        for x in range(1, 257):
            if time.time() > deadline:
                break
            for side in (0, 1):
                if x in best[side]:
                    continue
                candidate = best[side] + [x]
                if not _is_sidon(candidate):
                    continue
                d_new = _diffs(candidate)
                d_other = _diffs(best[1 - side])
                if d_new & d_other:
                    continue
                best[side] = candidate
                best_score += 1
                improved = True
                break
            if improved:
                break
        
        if improved:
            continue
        
        # Try swap: remove one element, add another
        for side in (0, 1):
            if time.time() > deadline:
                break
            for i in range(len(best[side])):
                if time.time() > deadline:
                    break
                removed = best[side][i]
                without = best[side][:i] + best[side][i+1:]
                for x in range(1, 257):
                    if x == removed or x in best[side]:
                        continue
                    candidate = without + [x]
                    if not _is_sidon(candidate):
                        continue
                    d_new = _diffs(candidate)
                    d_other = _diffs(best[1 - side])
                    if d_new & d_other:
                        continue
                    # Check that removing didn't break anything (it shouldn't, since we're removing)
                    best[side] = candidate
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_sets = [[], []]
    best_score = 0
    
    random.seed(42)
    
    # Phase 1: Random order greedy
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        order = list(range(1, 257))
        random.shuffle(order)
        
        sets = _greedy_construct(order, deadline)
        score = len(sets[0]) + len(sets[1])
        
        if score > best_score:
            best_score = score
            best_sets = [list(sets[0]), list(sets[1])]
        
        if time.time() < deadline - 5.0:
            improved = _local_improve(sets, deadline)
            score = len(improved[0]) + len(improved[1])
            if score > best_score:
                best_score = score
                best_sets = [list(improved[0]), list(improved[1])]
    
    # Phase 2: Cost-based greedy
    if time.time() < deadline - 15.0:
        for r in range(10):
            if time.time() > deadline - 10.0:
                break
            
            # Different random subsets/orders for cost-based
            available = list(range(1, 257))
            random.shuffle(available)
            
            sets = _greedy_cost_based(available, deadline)
            score = len(sets[0]) + len(sets[1])
            
            if score > best_score:
                best_score = score
                best_sets = [list(sets[0]), list(sets[1])]
            
            if time.time() < deadline - 5.0:
                improved = _local_improve(sets, deadline)
                score = len(improved[0]) + len(improved[1])
                if score > best_score:
                    best_score = score
                    best_sets = [list(improved[0]), list(improved[1])]
    
    # Phase 3: Biased orders
    if time.time() < deadline - 10.0:
        biases = [
            lambda x: x % 7,
            lambda x: (x * 3) % 11,
            lambda x: x % 13,
            lambda x: x % 5,
            lambda x: (x * 7) % 17,
            lambda x: x % 9,
        ]
        for bias_fn in biases:
            if time.time() > deadline - 5.0:
                break
            order = sorted(range(1, 257), key=bias_fn)
            
            sets = _greedy_construct(order, deadline)
            score = len(sets[0]) + len(sets[1])
            
            if score > best_score:
                best_score = score
                best_sets = [list(sets[0]), list(sets[1])]
            
            if time.time() < deadline - 3.0:
                improved = _local_improve(sets, deadline)
                score = len(improved[0]) + len(improved[1])
                if score > best_score:
                    best_score = score
                    best_sets = [list(improved[0]), list(improved[1])]
    
    # Final polish: try to add any remaining elements to best_sets
    if time.time() < deadline - 1.0:
        final = _local_improve(best_sets, deadline)
        if len(final[0]) + len(final[1]) > best_score:
            best_sets = final
            best_score = len(final[0]) + len(final[1])
    
    # Ensure nonempty
    if not best_sets[0]:
        best_sets[0] = [1]
    if not best_sets[1]:
        best_sets[1] = [2]
    
    return best_sets
