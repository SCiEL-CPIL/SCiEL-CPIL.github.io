import time
import random

def construct():
    deadline = time.time() + 50
    n = 256
    
    # Precompute conflicts
    conflicts = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    # Start with all odds
    s = set(range(1, n + 1, 2))
    
    # Greedily add evens in natural order
    evens = list(range(2, n + 1, 2))
    for e in evens:
        if not (conflicts[e] & s):
            s.add(e)
    
    best = list(s)
    best_size = len(s)
    
    # Try multiple random orderings
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 5:
            break
        s = set(range(1, n + 1, 2))
        evens_shuffled = list(range(2, n + 1, 2))
        random.shuffle(evens_shuffled)
        for e in evens_shuffled:
            if not (conflicts[e] & s):
                s.add(e)
        if len(s) > best_size:
            best = list(s)
            best_size = len(s)
    
    # Local search: try to add one more element by removing some
    s_set = set(best)
    for iteration in range(200):
        if time.time() > deadline - 3:
            break
        improved = False
        # Try adding each non-member
        for b in range(1, n + 1):
            if b in s_set:
                continue
            # Find which elements in s_set conflict with b
            conflicts_with_b = conflicts[b] & s_set
            if not conflicts_with_b:
                # Can add directly
                s_set.add(b)
                best = list(s_set)
                best_size = len(s_set)
                improved = True
                break
            # Try removing one conflicting element to add b
            for a in list(conflicts_with_b):
                new_set = s_set - {a} | {b}
                if len(new_set) > best_size:
                    # Verify validity
                    valid = True
                    for x in new_set:
                        if x != b and (conflicts[x] & (new_set - {x})):
                            # This shouldn't happen if we only removed a and added b
                            # and b doesn't conflict with new_set - {b}
                            pass
                    # Just check b against new_set - {b}
                    if not (conflicts[b] & (new_set - {b})):
                        s_set = new_set
                        best = list(s_set)
                        best_size = len(s_set)
                        improved = True
                        break
        if not improved:
            break
    
    return best
