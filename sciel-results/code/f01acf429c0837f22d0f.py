import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    # Perfect squares up to 512 (max sum is 256+256=512)
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Diagonal constraint: 2a must not be a perfect square
    # a = k^2 / 2 must be integer, so k must be even: k=2m, a=2m^2
    diagonal_bad = set()
    m = 1
    while 2 * m * m <= 256:
        diagonal_bad.add(2 * m * m)
        m += 1
    
    # Build adjacency: for each vertex, which other vertices conflict
    # conflict[i] = set of j where i+j is a perfect square (i != j)
    # Also self-conflict if 2*i is a perfect square (diagonal)
    n = 256
    conflicts = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        if a in diagonal_bad:
            continue  # can't be in set at all
        for b in range(a, n + 1):
            if b in diagonal_bad:
                continue
            if a + b in squares:
                if a != b:
                    conflicts[a].add(b)
                    conflicts[b].add(a)
                # if a==b, it's a self-conflict, already handled by diagonal_bad
    
    # Available vertices (not diagonal_bad)
    available = [i for i in range(1, n + 1) if i not in diagonal_bad]
    
    best_size = 0
    best_set = set()
    
    def is_valid(s):
        """Check if set s is valid (no pair sums to perfect square, including diagonal)"""
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                if lst[i] + lst[j] in squares:
                    return False
        return True
    
    def greedy_construct(order):
        """Greedy: add vertices in given order if no conflict with current set"""
        current = set()
        current_conflicts = set()
        for v in order:
            if v in diagonal_bad:
                continue
            # Check if v conflicts with any in current
            if not (current_conflicts & conflicts[v]):
                current.add(v)
                current_conflicts |= conflicts[v]
        return current
    
    def local_search(s, time_limit):
        """Improve set s using add/remove/swap moves"""
        current = set(s)
        current_size = len(current)
        current_conflicts = set()
        for v in current:
            current_conflicts |= conflicts[v]
        
        # Track which vertices are in set for quick lookup
        in_set = set(current)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try adding each available vertex not in set
            for v in available:
                if v in in_set:
                    continue
                # Can we add v?
                if not (current_conflicts & conflicts[v]):
                    current.add(v)
                    in_set.add(v)
                    current_conflicts |= conflicts[v]
                    current_size += 1
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one, add two? Or remove one add one?
                # Try: for each v in current, remove it and see if we can add more
                # Simpler: try removing each vertex and see if size improves
                for v in list(current):
                    # Remove v
                    current.discard(v)
                    in_set.discard(v)
                    # Recompute conflicts
                    new_conflicts = set()
                    for u in current:
                        new_conflicts |= conflicts[u]
                    
                    # Try to add back as many as possible
                    added_any = False
                    for w in available:
                        if w in in_set:
                            continue
                        if not (new_conflicts & conflicts[w]):
                            current.add(w)
                            in_set.add(w)
                            new_conflicts |= conflicts[w]
                            added_any = True
                    
                    if len(current) > current_size:
                        current_size = len(current)
                        current_conflicts = new_conflicts
                        improved = True
                        break
                    else:
                        # Restore v
                        current.add(v)
                        in_set.add(v)
                        current_conflicts = new_conflicts | conflicts[v]
        
        return current
    
    # Strategy 1: Greedy with different orderings
    random.seed(42)
    
    # Greedy by natural order
    s1 = greedy_construct(list(range(1, n + 1)))
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = set(s1)
    
    # Greedy by reverse order
    s2 = greedy_construct(list(range(n, 0, -1)))
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = set(s2)
    
    # Multiple random restarts with local search
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random permutation
        order = available[:]
        random.shuffle(order)
        
        # Greedy construction
        current = greedy_construct(order)
        
        if time.time() > deadline - 5:
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)
            break
        
        # Local search
        current = local_search(current, deadline - 2)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
        
        # Verify
        if not is_valid(best_set):
            # Should not happen, but safety check
            pass
    
    # Final verification
    if not is_valid(best_set):
        # Fallback: simple valid set
        fallback = set()
        for i in range(1, n + 1):
            if i in diagonal_bad:
                continue
            ok = True
            for j in fallback:
                if i + j in squares:
                    ok = False
                    break
            if ok:
                fallback.add(i)
        best_set = fallback
    
    return list(best_set)
