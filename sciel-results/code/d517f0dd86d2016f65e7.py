import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 192
    K = 3
    
    # Helper: Check if adding x to S is valid
    # S is a set of ints
    # x is an int not in S
    def is_valid_addition(S, x):
        # Check 1: x divides sum of any 3 distinct elements in S
        # S must have at least 3 elements
        if len(S) < 3:
            # Check 2: any element in S divides sum of x and 2 others
            # This is covered by the general check below if we treat S U {x}
            pass
        
        S_list = list(S)
        
        # Check if x divides sum of any 3 from S
        for t in combinations(S_list, 3):
            if sum(t) % x == 0:
                return False
        
        # Check if any y in S divides sum of x and 2 others from S\{y}
        # We need to check for each y in S, if y divides (x + a + b) for any a,b in S\{y}
        # To be efficient, we can precompute sums?
        # For now, simple loop.
        for y in S_list:
            others = [v for v in S_list if v != y]
            # We need to check if y divides x + a + b for any distinct a,b in others
            # Note: a and b must be distinct from y and each other.
            # Since y is removed from others, a,b are distinct from y.
            # a and b are distinct by combinations.
            # Also a,b must be distinct from x? No, x is new.
            # The summands are x, a, b. They must be distinct.
            # x is distinct from a,b because x not in S.
            # a,b distinct from y.
            # So we just need a,b distinct in others.
            for a, b in combinations(others, 2):
                if (x + a + b) % y == 0:
                    return False
        return True

    # Helper: Check if a set S is valid
    def is_valid_set(S):
        if len(S) < 4:
            return True
        S_list = list(S)
        for x in S_list:
            others = [v for v in S_list if v != x]
            for t in combinations(others, 3):
                if sum(t) % x == 0:
                    return False
        return True

    # Greedy initialization
    S = set()
    candidates = list(range(N, 0, -1))
    random.shuffle(candidates) # Randomize to avoid deterministic bad local minima
    
    # Actually, largest first is a good heuristic for validity, but random might find larger sets?
    # Let's try a mix: sort by value, but add random noise.
    # Or just start with a known good seed and improve.
    
    # Let's start with empty and add greedily with randomization
    # Or start with all numbers and remove?
    # Removing is often faster to find a valid set.
    
    # Strategy: Start with a large random set, prune to valid, then try to add back.
    
    # 1. Initialize S with a random subset of size 60
    S = set(random.sample(range(1, N+1), 60))
    
    # 2. Prune S to make it valid
    # While S is not valid, remove an element involved in a conflict
    while not is_valid_set(S) and time.time() < deadline - 10:
        # Find a conflict
        conflict_found = False
        S_list = list(S)
        random.shuffle(S_list) # Randomize removal order
        for x in S_list:
            if x not in S: continue
            others = [v for v in S if v != x]
            # Check if x divides any sum of 3 others
            for t in combinations(others, 3):
                if sum(t) % x == 0:
                    # Conflict found. Remove x? Or one of t?
                    # Removing x is safe.
                    S.remove(x)
                    conflict_found = True
                    break
            if conflict_found:
                break
        
        if not conflict_found:
            # No x divides sum of 3 others?
            # But is_valid_set checks all x.
            # If we didn't find one in the loop, maybe the loop finished without finding?
            # The loop breaks on first conflict.
            # If no conflict found for any x, then is_valid_set should return True.
            # Unless my is_valid_set and this loop are inconsistent.
            # Let's just re-check.
            if is_valid_set(S):
                break
            else:
                # Should not happen, but force break to avoid infinite loop
                break

    # Now S is valid.
    # 3. Try to add elements from E (excluded)
    E = set(range(1, N+1)) - S
    
    # Local Search
    best_size = len(S)
    
    while time.time() < deadline:
        if not E:
            break
            
        # Score each e in E by number of conflicts
        # To save time, only check a random subset of E if E is large
        # But E is at most 192.
        
        # Randomly pick a few candidates from E
        num_tries = 10
        tried = 0
        while tried < num_tries and time.time() < deadline:
            e = random.choice(list(E))
            tried += 1
            
            # Check if adding e is valid
            if is_valid_addition(S, e):
                S.add(e)
                E.remove(e)
                if len(S) > best_size:
                    best_size = len(S)
                # Continue trying to add more
            else:
                # Try to resolve conflict by swapping
                # Find conflicts caused by e
                # 1. e divides sum of 3 in S
                # 2. y in S divides e + a + b
                
                # This is complex to implement fully.
                # Simple heuristic: If adding e fails, just skip it for now.
                # To improve, we can try to remove a random element from S that is "large" or "conflicting"
                # and see if e fits.
                
                # Let's try a simple swap:
                # Pick a random y from S.
                # Remove y.
                # Check if S U {e} is valid.
                # If valid, accept swap (size same).
                # If not, put y back.
                
                # This might help escape local minima.
                if random.random() < 0.5: # 50% chance to try a swap
                    y = random.choice(list(S))
                    S.remove(y)
                    if is_valid_addition(S, e):
                        S.add(e)
                        E.remove(e)
                        E.add(y)
                        # Size unchanged, but composition changed
                    else:
                        S.add(y)
                        # Revert

    return list(S)
