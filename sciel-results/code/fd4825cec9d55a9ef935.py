import time
import random
from fractions import Fraction

def _calc_score(S):
    """Calculate exact sum of reciprocals as a float for comparison, 
    but keep track of exact fraction if needed? 
    The task says 'score is its floating serialization'. 
    So float sum is fine for internal comparison, 
    but we must return the list. 
    Using float sum is fast and precise enough for comparison 
    given the small N."""
    return sum(1.0 / x for x in S)

def _is_valid_add(S, y):
    """Check if y can be added to S without forming a 3-AP."""
    if y in S:
        return False
    # Check for 3-APs involving y
    # 1. y is midpoint: a, y, b -> 2y = a+b
    #    a+b = 2y. 
    #    We can check if there exists a in S such that 2y-a is in S.
    # 2. y is endpoint: a, b, y -> 2b = a+y -> b = (a+y)/2
    #    Or y, a, b -> 2a = y+b -> b = 2a-y
    
    # Efficient check:
    # For each a in S:
    #   Check if (a+y) is even. If so, mid = (a+y)//2. If mid in S and mid != a and mid != y, invalid.
    #   Check if 2*y - a is in S. Let b = 2*y - a. If b in S and b != a and b != y, invalid.
    #   Check if 2*a - y is in S. Let b = 2*a - y. If b in S and b != a and b != y, invalid.
    
    for a in S:
        # Case 1: a, y, b (y is midpoint) -> 2y = a + b => b = 2y - a
        b = 2 * y - a
        if b in S and b != a and b != y:
            return False
        
        # Case 2: a, b, y (b is midpoint) -> 2b = a + y => b = (a+y)/2
        if (a + y) % 2 == 0:
            b = (a + y) // 2
            if b in S and b != a and b != y:
                return False
        
        # Case 3: y, a, b (a is midpoint) -> 2a = y + b => b = 2a - y
        b = 2 * a - y
        if b in S and b != a and b != y:
            return False
            
    return True

def _greedy_init():
    S = set()
    for x in range(1, 257):
        if _is_valid_add(S, x):
            S.add(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initialize with greedy
    S = _greedy_init()
    best_S = set(S)
    best_score = _calc_score(best_S)
    
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.999
    
    # Precompute candidates? No, just pick randoms.
    all_elements = list(range(1, 257))
    
    while time.time() < deadline:
        if T < 0.001:
            break
            
        # Generate a move
        # 50% chance Add, 50% chance Swap
        if random.random() < 0.5:
            # Try to add a random element not in S
            candidates = [x for x in all_elements if x not in S]
            if not candidates:
                continue
            y = random.choice(candidates)
            if _is_valid_add(S, y):
                new_S = S | {y}
                new_score = _calc_score(new_S)
                delta = new_score - _calc_score(S)
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    S = new_S
                    if new_score > best_score:
                        best_score = new_score
                        best_S = new_S
        else:
            # Try to swap x in S with y not in S
            if not S:
                continue
            x = random.choice(list(S))
            candidates = [y for y in all_elements if y not in S]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # Check if swap is valid
            # Remove x, then check if y is valid in S \ {x}
            S_temp = S - {x}
            if _is_valid_add(S_temp, y):
                new_S = S_temp | {y}
                new_score = _calc_score(new_S)
                old_score = _calc_score(S)
                delta = new_score - old_score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    S = new_S
                    if new_score > best_score:
                        best_score = new_score
                        best_S = new_S
        
        T *= cooling_rate
        
    return list(best_S)
