import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s
    
    n = 256
    
    # Helper: check if adding x to set S creates a 3-AP
    # A 3-AP: a, b, c with a+c=2b, a<b<c
    # When adding x, check:
    # 1. x as middle: exists a,c in S with a+c=2x
    # 2. x as first: exists b,c in S with x+c=2b => c=2b-x, need c>x
    # 3. x as last: exists a,b in S with a+x=2b => a=2b-x, need a<x
    
    def can_add(S, x):
        # Check if x as middle: a+c=2x with a,c in S, a<c
        for a in S:
            c = 2*x - a
            if c != x and c in S and c > x:
                return False
        # Check if x as first term: x, b, c with x+c=2b => c=2b-x
        for b in S:
            if b > x:
                c = 2*b - x
                if c != x and c in S and c > b:
                    return False
        # Check if x as last term: a, b, x with a+x=2b => a=2b-x
        for b in S:
            if b < x:
                a = 2*b - x
                if a != x and a in S and a < b:
                    return False
        return True
    
    # Greedy initialization: add elements in order of decreasing 1/x (increasing x)
    # but skip those that would create 3-AP
    S = set()
    for x in range(1, n+1):
        if can_add(S, x):
            S.add(x)
    
    def reciprocal_sum(S):
        return sum(1.0/x for x in S)
    
    best_S = set(S)
    best_val = reciprocal_sum(S)
    
    # Local search: try adding/removing/swapping
    random.seed(42)
    
    # Precompute for speed: list of elements
    def try_improve(S):
        nonlocal best_S, best_val
        improved = False
        # Try adding each element not in S
        for x in range(1, n+1):
            if x not in S and can_add(S, x):
                new_S = S | {x}
                new_val = reciprocal_sum(new_S)
                if new_val > best_val:
                    best_val = new_val
                    best_S = new_S
                    S = new_S
                    improved = True
        # Try removing elements to allow adding others (this is more complex)
        return improved
    
    # Iterate: try to add elements greedily
    changed = True
    while changed and time.time() < deadline:
        changed = try_improve(S)
    
    # Simulated annealing phase
    S = set(best_S)
    current_val = reciprocal_sum(S)
    
    # Precompute all valid additions for current S
    def get_valid_additions(S):
        valid = []
        for x in range(1, n+1):
            if x not in S and can_add(S, x):
                valid.append(x)
        return valid
    
    temperature = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        valid_adds = get_valid_additions(S)
        if not valid_adds and len(S) > 1:
            # Try removing a random element and see if we can add something better
            removed = random.choice(list(S))
            S.remove(removed)
            valid_adds = get_valid_additions(S)
            if valid_adds:
                # Try to add the best available
                best_x = max(valid_adds, key=lambda x: 1.0/x)
                S.add(best_x)
            else:
                S.add(removed)  # put it back
            continue
        
        if valid_adds:
            # Pick an element to add, biased toward smaller x
            # Use temperature to occasionally pick worse options
            if temperature > 0.01 and random.random() < 0.3:
                x = random.choice(valid_adds)
            else:
                x = min(valid_adds)  # smallest x = largest 1/x
            S.add(x)
            new_val = reciprocal_sum(S)
            delta = new_val - current_val
            if delta > 0 or random.random() < __import__('math').exp(delta/temperature):
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_S = set(S)
            else:
                S.discard(x)
        
        temperature *= cooling
        if temperature < 0.001:
            temperature = 0.001
    
    return sorted(best_S)
