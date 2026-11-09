"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    N = 256
    
    def can_add(S, x):
        # Check if adding x to S creates a 3-term AP
        # Need to check: for any a, b in S, 2b - a != x and 2a - b != x
        # Actually, a 3-AP is (a, b, c) where a + c = 2b
        # So if we add x, we need to check:
        # 1. x as the middle: a + c = 2x, so for any a in S, c = 2x - a must not be in S
        # 2. x as the end: a + x = 2b, so for any a in S, b = (a + x) / 2 must not be in S (and must be integer)
        # 3. x as the other end: x + c = 2b, so for any c in S, b = (x + c) / 2 must not be in S (and must be integer)
        # More efficiently: for each b in S, check if 2b - x is in S (x would be the end)
        # and for each a in S, check if (a + x) / 2 is in S (x would be the end, a the other end)
        S_set = S  # S is already a set
        # Check if x can be the middle term: need a, c in S such that a + c = 2x
        # i.e., for any a in S, 2x - a in S
        for a in S_set:
            c = 2 * x - a
            if c in S_set and c != a:
                return False
        # Check if x can be an end term: need a, b in S such that a + x = 2b
        # i.e., for any a in S, (a + x) / 2 in S and (a + x) is even
        for a in S_set:
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in S_set and b != a and b != x:
                    return False
        return True
    
    def reciprocal_sum(S):
        return sum(1.0 / x for x in S)
    
    # Greedy construction: add elements in order of largest reciprocal (smallest x)
    # that don't create a 3-AP
    S = set()
    for x in range(1, N + 1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_score = reciprocal_sum(best_S)
    
    # Local search with add/remove/swap moves
    random.seed(42)
    
    def try_improve(S, current_score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements not in S
            for x in range(1, N + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
                    new_score = current_score + 1.0 / x
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
            
            # Try removing elements and adding others
            if not improved and time.time() < deadline:
                for x in list(S):
                    S_copy = set(S)
                    S_copy.remove(x)
                    # Try adding elements to S_copy
                    for y in range(1, N + 1):
                        if y not in S_copy and can_add(S_copy, y):
                            S_copy.add(y)
                            new_score = current_score - 1.0 / x + 1.0 / y
                            if new_score > current_score:
                                S = S_copy
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
        
        return S, current_score
    
    best_S, best_score = try_improve(set(best_S), best_score)
    
    # Further refinement: try multiple restarts with different random orders
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Randomized greedy: shuffle order but prioritize small elements
        S = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        # Sort by reciprocal (smallest x first) with some randomization
        order.sort(key=lambda x: (x + random.random() * 10))
        
        for x in order:
            if x <= N and can_add(S, x):
                S.add(x)
        
        S, score = try_improve(S, reciprocal_sum(S))
        
        if score > best_score:
            best_S = S
            best_score = score
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
