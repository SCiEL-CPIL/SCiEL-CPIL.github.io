import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 128
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                ab = a + b
                ac = a + c
                bc = b + c
                if ab <= N and ac <= N and bc <= N:
                    triples.append((a, b, c, ab, ac, bc))
    
    triples_involving = [[] for _ in range(N + 1)]
    for i, t in enumerate(triples):
        for val in t:
            if 1 <= val <= N:
                triples_involving[val].append(i)

    def is_valid_add(S, x):
        for idx in triples_involving[x]:
            t = triples[idx]
            valid = True
            for val in t:
                if val == x:
                    continue
                if val not in S:
                    valid = False
                    break
            if valid:
                return False
        return True

    # Start with all odds
    S = set(range(1, N + 1, 2))
    best_S = set(S)
    best_size = len(S)
    
    random.seed(42)
    
    # Phase 1: Greedy addition
    # Try to add any element
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for x in range(1, N + 1):
            if x in S:
                continue
            if is_valid_add(S, x):
                S.add(x)
                changed = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)

    # Phase 2: Local Search / SA on valid states
    # Since we are in valid states, we only move to valid states.
    # This is a random walk on the space of valid sets.
    # To escape local maxima, we can remove an element (valid) and then try to add others.
    
    T = 1.0
    while time.time() < deadline - 5:
        # Randomly pick an element to flip
        x = random.randint(1, N)
        
        if x in S:
            # Remove x
            # S is valid, so removing x keeps it valid.
            S.remove(x)
            # Now try to add a different element
            # Pick a random candidate to add
            candidates = [y for y in range(1, N + 1) if y not in S]
            if candidates:
                y = random.choice(candidates)
                if is_valid_add(S, y):
                    S.add(y)
        else:
            # Add x
            if is_valid_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Final greedy fill
    if time.time() < deadline - 2:
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x in best_S:
                    continue
                if is_valid_add(best_S, x):
                    best_S.add(x)
                    changed = True

    result = [int(x) for x in best_S]
    return result
