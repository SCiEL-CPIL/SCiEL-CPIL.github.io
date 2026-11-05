"""Combined program: fast integer LCM-based dissociation check + incremental backtracking search."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_dissociated_int(S):
    """Fast check: all subset sums of {1/x for x in S} distinct, via integer LCM."""
    if not S:
        return True
    L = 1
    for x in S:
        L = math.lcm(L, x)
    vals = [L // x for x in S]
    sums = {0}
    for v in vals:
        new = {s + v for s in sums}
        if new & sums:
            return False
        sums |= new
    return True

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill
    universe = list(range(1, 23))
    
    best = [1] # Fallback valid set
    best_size = 1
    
    def backtrack(current, remaining, current_lcm, current_sums, current_size):
        nonlocal best, best_size
        
        if time.time() > deadline - 2:
            return
        
        # Prune: even if we add all remaining, can we beat best?
        if current_size + len(remaining) <= best_size:
            return
        
        if not remaining:
            if current_size > best_size:
                best = list(current)
                best_size = current_size
            return
        
        x = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include x
        new_lcm = math.lcm(current_lcm, x)
        scale = new_lcm // current_lcm
        delta = new_lcm // x
        
        valid = True
        if delta % scale == 0:
            diff = delta // scale
            # Check if diff is in {s1 - s2 for s1, s2 in current_sums}
            # Equivalent to: exists s in current_sums such that s + diff in current_sums
            # Note: s1 - s2 = diff => s1 = s2 + diff
            if any((s + diff) in current_sums for s in current_sums):
                valid = False
        
        if valid:
            # Construct new sums
            if scale == 1:
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s)
                    new_sums.add(s + delta)
            else:
                new_sums = set()
                for s in current_sums:
                    s_scaled = s * scale
                    new_sums.add(s_scaled)
                    new_sums.add(s_scaled + delta)
            
            current.append(x)
            backtrack(current, rest, new_lcm, new_sums, current_size + 1)
            current.pop()
        
        # Option 2: exclude x
        backtrack(current, rest, current_lcm, current_sums, current_size)

    # Try different orderings to find large sets quickly
    # 1. Descending
    order1 = sorted(universe, reverse=True)
    backtrack([], order1, 1, {0}, 0)
    
    if time.time() > deadline - 10:
        if _is_dissociated_int(best):
            return sorted(best)
        return [1]
        
    # 2. Ascending
    order2 = sorted(universe)
    backtrack([], order2, 1, {0}, 0)
    
    if time.time() > deadline - 10:
        if _is_dissociated_int(best):
            return sorted(best)
        return [1]
        
    # 3. Primes first
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
        
    primes = [x for x in range(1, 23) if is_prime(x)]
    composites = [x for x in range(1, 23) if not is_prime(x)]
    order3 = primes + composites
    backtrack([], order3, 1, {0}, 0)
    
    if time.time() > deadline - 5:
        if _is_dissociated_int(best):
            return sorted(best)
        return [1]
        
    # 4. Random shuffles
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 2:
            break
        order4 = list(universe)
        random.shuffle(order4)
        backtrack([], order4, 1, {0}, 0)
        
    if not best or not _is_dissociated_int(best):
        return [1]
        
    return sorted(best)
# EVOLVE-BLOCK-END
