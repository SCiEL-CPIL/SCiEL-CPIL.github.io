"""Combined program: fast integer LCM-based dissociation check + backtracking search."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill
    universe = list(range(1, 23))
    
    # Known valid 9-element set as fallback
    base = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = list(base)
    
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    
    # Fast dissociation check using integer LCM
    def is_dissociated(S):
        if not S:
            return True
        L = 1
        for x in S:
            L = math.lcm(L, x)
        vals = [L // x for x in S]
        sums = {0}
        for v in vals:
            new = {s + v for s in sums}
            if len(new) != len(sums):
                return False
            sums |= new
        return True
    
    # Backtracking with incremental state
    # State: current set, current LCM, current subset sums
    def backtrack(current, current_L, current_sums, remaining, current_size):
        nonlocal best
        if time.time() > deadline - 2:
            return
        
        # Prune
        if current_size + len(remaining) <= len(best):
            return
        
        if not remaining:
            if current_size > len(best):
                best = list(current)
            return
        
        x = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include x
        # New LCM = lcm(current_L, x)
        new_L = math.lcm(current_L, x)
        # New sums = current_sums ∪ {s + new_L//x for s in current_sums}
        v = new_L // x
        new_sums = current_sums | {s + v for s in current_sums}
        
        if len(new_sums) == len(current_sums) * 2:
            # No collision
            current.append(x)
            backtrack(current, new_L, new_sums, rest, current_size + 1)
            current.pop()
        
        # Option 2: exclude x
        backtrack(current, current_L, current_sums, rest, current_size)
    
    # Try different orderings
    orderings = [
        list(range(1, 23)),           # ascending
        list(range(22, 0, -1)),       # descending
        [x for x in range(1, 23) if is_prime(x)] + [x for x in range(1, 23) if not is_prime(x)],  # primes first
    ]
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        backtrack([], 1, {0}, order, 0)
    
    # Randomized greedy
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 23))
        random.shuffle(order)
        current = []
        current_L = 1
        current_sums = {0}
        for x in order:
            if time.time() > deadline - 2:
                break
            new_L = math.lcm(current_L, x)
            v = new_L // x
            new_sums = current_sums | {s + v for s in current_sums}
            if len(new_sums) == len(current_sums) * 2:
                current.append(x)
                current_L = new_L
                current_sums = new_sums
        if len(current) > len(best):
            best = current
    
    # Try to extend best by adding any remaining element
    if time.time() < deadline - 3:
        for x in range(1, 23):
            if x in best:
                continue
            if is_dissociated(best + [x]):
                best.append(x)
    
    # Final verification
    if not is_dissociated(best):
        return sorted(base)
    
    return sorted(best)
# EVOLVE-BLOCK-END
