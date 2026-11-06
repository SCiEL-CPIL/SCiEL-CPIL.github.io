import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin within 90s
    
    N = 256
    # Start with all odd numbers (128 elements)
    S = set(range(1, N + 1, 2))
    
    # Helper: check if x can be added to set S
    def can_add(x, S):
        for y in S:
            if (x * y) % (x + y) == 0:
                return False
        return True
    
    # Greedily add even numbers in various orders
    evens = list(range(2, N + 1, 2))
    
    # Try multiple orders to find a good starting point
    best_S = set(S)
    best_size = len(S)
    
    # Order 1: natural order
    for e in evens:
        if time.time() > deadline - 5:
            break
        if can_add(e, best_S):
            best_S.add(e)
    
    # Order 2: reversed
    S2 = set(S)
    for e in reversed(evens):
        if time.time() > deadline - 5:
            break
        if can_add(e, S2):
            S2.add(e)
    if len(S2) > len(best_S):
        best_S = S2
    
    # Order 3: shuffled (a few tries)
    for trial in range(5):
        if time.time() > deadline - 5:
            break
        S3 = set(S)
        evens_shuffled = evens[:]
        random.shuffle(evens_shuffled)
        for e in evens_shuffled:
            if can_add(e, S3):
                S3.add(e)
        if len(S3) > len(best_S):
            best_S = S3
    
    # Local search: try swapping elements to improve
    # If we can't add x, try removing some y that conflicts with x, then add x
    # But we want to maximize size, so swaps only help if we can add more later
    
    # Better local search: try to find a set of elements to remove that allows adding more
    # Simpler: try individual removals and see if we can add more evens afterward
    
    current_S = set(best_S)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try removing each element and see if we can add more
        for y in list(current_S):
            if time.time() > deadline - 2:
                break
            test_S = current_S - {y}
            # Try to add evens not in test_S
            added = 0
            for e in evens:
                if e in test_S:
                    continue
                if can_add(e, test_S):
                    test_S.add(e)
                    added += 1
            if added > 1:  # net gain
                current_S = test_S
                improved = True
                break
    
    best_S = current_S
    
    # Final local search: try swapping one element for another to enable additions
    for _ in range(3):
        if time.time() > deadline - 2:
            break
        improved = False
        for y in list(best_S):
            if time.time() > deadline - 2:
                break
            test_S = best_S - {y}
            added = 0
            for e in evens:
                if e in test_S:
                    continue
                if can_add(e, test_S):
                    test_S.add(e)
                    added += 1
            if added >= 1:
                best_S = test_S
                improved = True
                break
        if not improved:
            break
    
    return list(best_S)
