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

def _greedy_construct(order, deadline):
    """Greedy construction given an order of candidates."""
    sets = [[], []]
    diffs = [set(), set()]
    sums = [set(), set()]
    
    for x in order:
        if time.time() > deadline:
            break
        for side in (0, 1):
            # Check Sidon condition
            ok = True
            for a in sets[side]:
                s = a + x
                if s in sums[side]:
                    ok = False
                    break
                if a == x:
                    break  # a+a already checked via a==x
            if not ok:
                continue
            # Also need to check x+x
            if x + x in sums[side]:
                continue
            
            # Check difference disjointness
            new_diffs = set()
            for a in sets[side]:
                d = abs(x - a)
                new_diffs.add(d)
            
            if new_diffs & diffs[1 - side]:
                continue
            
            # Accept
            sets[side].append(x)
            for a in sets[side]:
                sums[side].add(a + x)
            sums[side].add(x + x)
            diffs[side].update(new_diffs)
    
    return sets

def _local_improve(sets, deadline):
    """Try to add more elements or swap elements to improve combined size."""
    best = [list(sets[0]), list(sets[1])]
    best_score = len(best[0]) + len(best[1])
    
    for _ in range(50):
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
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    best_sets = [[], []]
    best_score = 0
    
    # Multiple random restarts
    random.seed(42)
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a random order
        order = list(range(1, 257))
        random.shuffle(order)
        
        # Greedy construction
        sets = _greedy_construct(order, deadline)
        score = len(sets[0]) + len(sets[1])
        
        if score > best_score:
            best_score = score
            best_sets = [list(sets[0]), list(sets[1])]
        
        # Local improvement
        if time.time() < deadline - 5.0:
            improved = _local_improve(sets, deadline)
            score = len(improved[0]) + len(improved[1])
            if score > best_score:
                best_score = score
                best_sets = [list(improved[0]), list(improved[1])]
    
    # Try a few more targeted strategies if time remains
    if time.time() < deadline - 10.0:
        # Try with biased orders (prefer smaller numbers first, etc.)
        for bias in range(5):
            if time.time() > deadline:
                break
            order = list(range(1, 257))
            # Apply different biases
            if bias == 0:
                order = sorted(order, key=lambda x: x % 7)
            elif bias == 1:
                order = sorted(order, key=lambda x: (x * 3) % 11)
            elif bias == 2:
                order = sorted(order, key=lambda x: x % 13)
            elif bias == 3:
                random.shuffle(order)
                order = sorted(order, key=lambda x: x % 5)
            else:
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
    
    # Ensure nonempty
    if not best_sets[0]:
        best_sets[0] = [1]
    if not best_sets[1]:
        best_sets[1] = [2]
    
    return best_sets
