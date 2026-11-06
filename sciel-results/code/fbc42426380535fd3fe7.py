import random
import time

def construct():
    N = 512
    deadline = time.time() + 55.0  # 55s guard, leave 5s safety margin for 60s limit
    
    # Precompute all 4-term APs for fast conflict checking
    # A 4-term AP is (a, a+d, a+2d, a+3d) with d >= 1
    # For each element x, precompute which other elements it conflicts with
    # Actually, we need to check if adding x creates a 4-AP with existing elements
    
    # Better approach: for each x, precompute the set of "forbidden triples" 
    # that would complete a 4-AP if x were added
    
    # Let's use a different strategy: 
    # 1. Start with a known good construction (base-3 / no-zero-digits)
    # 2. Use local search / simulated annealing to improve
    
    # Base-3 construction: numbers with no '0' in base-3 representation
    def base3_no_zero(x):
        while x > 0:
            if x % 3 == 0:
                return False
            x //= 3
        return x == 0 or True  # 0 is not in our range, but handle edge case
    
    # Initial set from base-3 construction
    initial_set = set()
    for x in range(1, N + 1):
        # Check base-3 digits
        temp = x
        valid = True
        while temp > 0:
            if temp % 3 == 0:
                valid = False
                break
            temp //= 3
        if valid:
            initial_set.add(x)
    
    # Verify no 4-AP in initial set
    def has_4ap(S):
        S = set(S)
        for a in S:
            for d in range(1, (N - a) // 3 + 1):
                if (a + d in S) and (a + 2*d in S) and (a + 3*d in S):
                    return True
        return False
    
    # Simulated annealing approach
    def can_add(x, S):
        # Check if adding x to S creates a 4-AP
        # x could be any of the 4 terms
        # Case 1: x is first term: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        # Case 2: x is second term: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        # Case 3: x is third term: x-2d, x-d, x, x+d
        max_d3 = (x - 1) // 2
        for d in range(1, max_d3 + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        # Case 4: x is fourth term: x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        return True
    
    def can_remove(x, S):
        # Always possible to remove
        return True
    
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Try to improve by adding elements
    def try_add_elements(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
        return S
    
    best_set = try_add_elements(best_set)
    best_size = len(best_set)
    
    # Simulated annealing for further improvement
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Temperature schedule
    T = 10.0
    cooling = 0.995
    
    while time.time() < deadline:
        if T < 0.01:
            T = 10.0  # Restart with higher temperature
        
        # Pick a random move
        move_type = random.random()
        if move_type < 0.5:
            # Try to add a random element
            x = random.randint(1, N)
            if x not in current_set:
                if can_add(x, current_set):
                    current_set.add(x)
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                else:
                    # Try removing one element from the conflicting AP and adding x
                    # This is complex, so skip for now
                    pass
        elif move_type < 0.8:
            # Try to remove a random element
            if current_set:
                x = random.choice(list(current_set))
                current_set.remove(x)
        else:
            # Try to swap: remove one, add another
            if current_set:
                x_remove = random.choice(list(current_set))
                x_add = random.randint(1, N)
                if x_add not in current_set and x_add != x_remove:
                    # Check if we can do the swap
                    temp_set = current_set - {x_remove}
                    if can_add(x_add, temp_set):
                        temp_set.add(x_add)
                        # Accept with some probability
                        delta = len(temp_set) - len(current_set)
                        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                            current_set = temp_set
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = set(current_set)
        
        T *= cooling
    
    return sorted(best_set)
