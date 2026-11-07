import time
import random

def construct():
    deadline = time.time() + 80.0
    
    def can_add(S, x):
        """
        Check if x can be added to S without violating the constraint.
        Constraint: No distinct a,b,c in S' such that a+b, a+c, b+c in S'.
        If we add x to S, the only new triples must involve x.
        So we check triples {x, a, b} where a, b in S, a != b.
        Conditions for {x, a, b} to be forbidden:
        1. x + a in S' (which is S since x+a != x)
        2. x + b in S' (which is S since x+b != x)
        3. a + b in S' (which is S U {x}, so a+b in S OR a+b == x)
        """
        # Find all a in S such that a+x is in S
        # These are the only candidates for 'a' in the triple
        candidates = [a for a in S if a + x in S]
        
        # Check pairs from candidates
        # We need a != b
        n = len(candidates)
        for i in range(n):
            a = candidates[i]
            for j in range(i + 1, n):
                b = candidates[j]
                # Check third sum
                if a + b in S or a + b == x:
                    return False
        return True

    def fill_set(S):
        """
        Greedily add any element from 1..128 not in S that can be added.
        Repeat until no more can be added.
        """
        improved = True
        while improved:
            improved = False
            # Iterate in a fixed order, or random? Fixed is deterministic for this step.
            # We can try 1..128
            for x in range(1, 129):
                if x in S:
                    continue
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    # Don't break, keep checking others in same pass?
                    # Or break and restart? Restarting is safer for finding more.
                    # But iterating through all is fine as we check against current S.
                    # Actually, adding one might block another.
                    # Let's just do one full pass, and if any added, do another.
        return S

    def greedy_build(base_set, order):
        S = set(base_set)
        for x in order:
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        # Fill remaining
        S = fill_set(S)
        return S

    # Initial base set: 63..128
    base_set = set(range(63, 129))
    
    best_set = set(base_set)
    
    # Try different orders for the "small" numbers 1..62
    small_nums = list(range(1, 63))
    
    # Order 1: Ascending
    S1 = greedy_build(base_set, small_nums)
    if len(S1) > len(best_set):
        best_set = S1
        
    # Order 2: Descending
    S2 = greedy_build(base_set, small_nums[::-1])
    if len(S2) > len(best_set):
        best_set = S2
        
    # Order 3-10: Random orders
    random.seed(42)
    for i in range(10):
        if time.time() > deadline - 10:
            break
        random.shuffle(small_nums)
        Si = greedy_build(base_set, small_nums)
        if len(Si) > len(best_set):
            best_set = Si

    # Local Search on the best set found
    # Strategy: Try to remove an element y, then fill. If size increases, keep it.
    # Also try swapping? 
    # Let's do a "perturb and fill" local search.
    
    current_set = set(best_set)
    
    while time.time() < deadline - 2:
        improved = False
        # Try removing random elements
        # We want to see if removing 1 allows adding 2.
        # Pick a random y from current_set
        y = random.choice(list(current_set))
        test_set = current_set - {y}
        test_set = fill_set(test_set)
        
        if len(test_set) > len(current_set):
            current_set = test_set
            improved = True
            # Update best if better
            if len(current_set) > len(best_set):
                best_set = set(current_set)
        
        # Also try simple "add" again in case fill_set missed something due to order?
        # fill_set is exhaustive for additions.
        
        # Try a specific swap: Remove y, Add x?
        # fill_set will add x if it's valid.
        
        # If no improvement after many tries, maybe try a different base?
        # Or just keep going.
        
    return list(best_set)

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify
    S = set(res)
    valid = True
    triples = 0
    for a in S:
        for b in S:
            if a == b: continue
            for c in S:
                if a == c or b == c: continue
                if a+b in S and a+c in S and b+c in S:
                    valid = False
                    triples += 1
    print(f"Valid: {valid}, Forbidden Triples: {triples}")
