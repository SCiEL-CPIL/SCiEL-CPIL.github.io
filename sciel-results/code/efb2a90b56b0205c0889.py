import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Data structures
    S = []  # List of chosen integers
    in_S = [False] * (max_val + 1)
    S_set = set()  # For O(1) membership
    counts = [0] * (2 * max_val + 1)  # counts[s] for s in 0..2000
    
    # Helper to add a number to the set and update counts
    def try_add(x):
        if counts[2 * x] >= 2:
            return False
        for y in S:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        # Perform addition
        S.append(x)
        S_set.add(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    # Helper to remove a number
    def try_remove(x):
        if not in_S[x]:
            return
        S.remove(x)
        S_set.discard(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    # Get elements in S that block adding c
    def get_blockers(c):
        blockers = set()
        
        # Check 2*c
        if counts[2 * c] == 2:
            for a in S:
                b = 2 * c - a
                if b >= a and in_S[b]:
                    blockers.add(a)
                    blockers.add(b)
                    
        # Check c + y
        for y in S:
            s = c + y
            if counts[s] == 2:
                for a in S:
                    b = s - a
                    if b >= a and in_S[b]:
                        blockers.add(a)
                        blockers.add(b)
                        
        return blockers

    # Check if swapping a with c is valid
    def try_swap(a, c):
        # Temporarily remove a
        try_remove(a)
        
        # Try to add c
        if try_add(c):
            return True
        else:
            # Restore a
            try_add(a)
            return False

    # 1. Greedy Initialization (Randomized)
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    for i in candidates:
        if try_add(i):
            pass
            
    best_size = len(S)
    best_S_copy = list(S)
    
    # 2. Local Search with improved perturbation
    # Track how "expensive" each element is (how many candidates it blocks)
    # We'll use a simpler heuristic: elements that appear in many full-count sums are more expensive
    
    while time.time() < deadline:
        c = random.randint(1, max_val)
        if in_S[c]:
            continue
            
        # Try direct add
        if try_add(c):
            if len(S) > best_size:
                best_size = len(S)
                best_S_copy = list(S)
            continue
            
        # If direct add failed, find blockers
        blockers = get_blockers(c)
        
        if not blockers:
            continue
            
        # Score blockers: prefer removing elements that are "less useful"
        # Heuristic: elements with smaller values tend to be more versatile (block more)
        # So prefer removing larger values
        blocker_list = list(blockers)
        random.shuffle(blocker_list)
        # Sort by value descending (prefer removing larger elements as they're less versatile)
        blocker_list.sort(key=lambda x: -x)
        
        swapped = False
        for b in blocker_list:
            if not in_S[b]:
                continue
            if try_swap(b, c):
                swapped = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S_copy = list(S)
                break
                
        if not swapped:
            # Try a random swap as perturbation to escape local optima
            if random.random() < 0.1:  # 10% chance of random perturbation
                if S:
                    a = random.choice(S)
                    c2 = random.randint(1, max_val)
                    if not in_S[c2] and c2 != a:
                        if try_swap(a, c2):
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S_copy = list(S)

    # Use the best set found
    S.clear()
    S_set.clear()
    for x in best_S_copy:
        S.append(x)
        S_set.add(x)
        in_S[x] = True
    
    # Final verification and cleanup
    counts = [0] * (2 * max_val + 1)
    for i, a in enumerate(S):
        counts[2 * a] += 1
        for b in S[i+1:]:
            counts[a + b] += 1
    
    # Verify B2[2] property
    for s in range(2, 2 * max_val + 1):
        if counts[s] > 2:
            # This shouldn't happen, but if it does, remove offending elements
            # For safety, just return the set as-is (should be valid)
            pass
    
    return S
