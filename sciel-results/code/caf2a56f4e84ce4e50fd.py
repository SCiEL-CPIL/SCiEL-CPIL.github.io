import numpy as np
import time
import random

def construct():
    n = 512
    start_time = time.time()
    max_time = 80.0  # Safety margin for 90s limit

    def is_valid_addition(x, S, n):
        # S is a boolean numpy array of length n+1
        # Check 4 cases for any d
        # Case 1: x is 4th term: x-3d, x-2d, x-d in S
        for d in range(1, (x-1)//3 + 1):
            if S[x-d] and S[x-2*d] and S[x-3*d]:
                return False
        
        # Case 2: x is 1st term: x+d, x+2d, x+3d in S
        for d in range(1, (n-x)//3 + 1):
            if S[x+d] and S[x+2*d] and S[x+3*d]:
                return False
        
        # Case 3: x is 2nd term: x-d, x+d, x+2d in S
        max_d = min(x-1, (n-x)//2)
        for d in range(1, max_d + 1):
            if S[x-d] and S[x+d] and S[x+2*d]:
                return False
        
        # Case 4: x is 3rd term: x-2d, x-d, x+d in S
        max_d = min((x-1)//2, n-x)
        for d in range(1, max_d + 1):
            if S[x-2*d] and S[x-d] and S[x+d]:
                return False
        
        return True

    # Base-3 digit-based construction: select numbers with no digit '2' in base 3
    # This creates a Cantor-like set that is 4-AP-free
    def has_no_two_in_base3(x):
        while x > 0:
            if x % 3 == 2:
                return False
            x //= 3
        return True
    
    # Build the Cantor-like set
    cantor_set = set()
    for i in range(1, n+1):
        if has_no_two_in_base3(i):
            cantor_set.add(i)
    
    # Convert to boolean array
    S_cantor = np.zeros(n+1, dtype=bool)
    for i in cantor_set:
        S_cantor[i] = True
    
    # Verify the Cantor set is valid (should be, but let's check)
    # For each element, check if it forms a 4-AP with others
    def verify_set(S, n):
        """Verify that S contains no 4-AP"""
        for x in range(1, n+1):
            if not S[x]:
                continue
            # Check if x can be part of a 4-AP with other elements in S
            # We need to check all 4 positions
            # Case 1: x is 4th term
            for d in range(1, (x-1)//3 + 1):
                if S[x-d] and S[x-2*d] and S[x-3*d]:
                    return False, x, d, "4th"
            # Case 2: x is 1st term
            for d in range(1, (n-x)//3 + 1):
                if S[x+d] and S[x+2*d] and S[x+3*d]:
                    return False, x, d, "1st"
            # Case 3: x is 2nd term
            max_d = min(x-1, (n-x)//2)
            for d in range(1, max_d + 1):
                if S[x-d] and S[x+d] and S[x+2*d]:
                    return False, x, d, "2nd"
            # Case 4: x is 3rd term
            max_d = min((x-1)//2, n-x)
            for d in range(1, max_d + 1):
                if S[x-2*d] and S[x-d] and S[x+d]:
                    return False, x, d, "3rd"
        return True, None, None, None
    
    # The Cantor set should be valid by construction
    # Let's also try the standard greedy for comparison
    S_greedy = np.zeros(n+1, dtype=bool)
    for x in range(1, n+1):
        if is_valid_addition(x, S_greedy, n):
            S_greedy[x] = True
    
    # Start with the better of the two
    cantor_count = int(np.sum(S_cantor))
    greedy_count = int(np.sum(S_greedy))
    
    if cantor_count >= greedy_count:
        S_best = S_cantor.copy()
        best_count = cantor_count
    else:
        S_best = S_greedy.copy()
        best_count = greedy_count
    
    best_set_list = [i for i in range(1, n+1) if S_best[i]]
    
    # Local search: try to add elements by removing conflicting ones
    # For each missing element, find which elements in S conflict with it
    # Then try removing one conflicting element and adding the new one
    
    def find_conflicts(x, S, n):
        """Find elements in S that form a 4-AP with x"""
        conflicts = set()
        # Case 1: x is 4th term
        for d in range(1, (x-1)//3 + 1):
            if S[x-d] and S[x-2*d] and S[x-3*d]:
                conflicts.update([x-d, x-2*d, x-3*d])
        # Case 2: x is 1st term
        for d in range(1, (n-x)//3 + 1):
            if S[x+d] and S[x+2*d] and S[x+3*d]:
                conflicts.update([x+d, x+2*d, x+3*d])
        # Case 3: x is 2nd term
        max_d = min(x-1, (n-x)//2)
        for d in range(1, max_d + 1):
            if S[x-d] and S[x+d] and S[x+2*d]:
                conflicts.update([x-d, x+d, x+2*d])
        # Case 4: x is 3rd term
        max_d = min((x-1)//2, n-x)
        for d in range(1, max_d + 1):
            if S[x-2*d] and S[x-d] and S[x+d]:
                conflicts.update([x-2*d, x-d, x+d])
        return conflicts
    
    # Randomized restarts with local search
    for restart in range(10000):
        if time.time() - start_time > max_time:
            break
        
        # Start from a random perturbation of the best set
        S_curr = S_best.copy()
        
        # Randomly remove some elements to create room
        num_remove = random.randint(1, 5)
        elements_in_set = [i for i in range(1, n+1) if S_curr[i]]
        if len(elements_in_set) <= num_remove:
            continue
        to_remove = random.sample(elements_in_set, num_remove)
        for elem in to_remove:
            S_curr[elem] = False
        
        # Try to add back elements and more
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        for x in candidates:
            if not S_curr[x] and is_valid_addition(x, S_curr, n):
                S_curr[x] = True
        
        curr_count = int(np.sum(S_curr))
        
        if curr_count > best_count:
            best_count = curr_count
            S_best = S_curr.copy()
            best_set_list = [i for i in range(1, n+1) if S_best[i]]
    
    # Final local search: try to add any missing element
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for x in range(1, n+1):
            if not S_best[x] and is_valid_addition(x, S_best, n):
                S_best[x] = True
                best_count += 1
                improved = True
    
    best_set_list = [i for i in range(1, n+1) if S_best[i]]
    
    return best_set_list
