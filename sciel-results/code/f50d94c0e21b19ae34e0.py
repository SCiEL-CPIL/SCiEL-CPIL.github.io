"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    n = 192
    all_nums = list(range(1, n + 1))
    
    def can_add(x, S):
        # S is a list
        # 1. x divides sum of 3 in S
        for a, b, c in combinations(S, 3):
            if (a + b + c) % x == 0:
                return False
        # 2. a in S divides sum of {x, b, c}
        # Iterate a in S
        for a in S:
            # Check pairs b, c in S \ {a}
            # To avoid creating a new list every time, we can just check
            for i in range(len(S)):
                if S[i] == a:
                    continue
                for j in range(i + 1, len(S)):
                    if S[j] == a:
                        continue
                    if (x + S[i] + S[j]) % a == 0:
                        return False
        return True

    # Greedy: Largest first
    S = []
    for x in range(n, 0, -1):
        if can_add(x, S):
            S.append(x)
            
    best_S = list(S)
    best_size = len(S)
    
    # Local Search
    unselected = set(all_nums) - set(S)
    
    # Try to improve best_S
    S = list(best_S)
    unselected = set(all_nums) - set(S)
    
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        if not unselected:
            break
            
        # Randomly pick a candidate to add
        v = random.choice(list(unselected))
        
        if can_add(v, S):
            S.append(v)
            unselected.remove(v)
            if len(S) > best_size:
                best_S = list(S)
                best_size = len(S)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Try to swap: remove u, add v
            # Find a u in S such that can_add(v, S \ {u}) is True
            # To save time, only try a few random u's or the first one found
            # We'll try up to 5 random u's
            swapped = False
            for _ in range(5):
                if not S:
                    break
                u = random.choice(S)
                S.remove(u)
                if can_add(v, S):
                    S.append(v)
                    unselected.add(u)
                    unselected.remove(v)
                    swapped = True
                    break
                else:
                    # Revert
                    S.append(u)
                    # Note: unselected was not modified yet
            
            if not swapped:
                no_improve_count += 1
                
            # If we swapped, size is same.
            # We don't update best_S unless size increases.
            # But we continue searching from this new state.

        # If stuck for a long time, maybe restart?
        # For now, just keep going.
        
    return best_S
# EVOLVE-BLOCK-END
