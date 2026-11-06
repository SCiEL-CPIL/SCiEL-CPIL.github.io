"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 256
    
    # Start with upper tail as safe base
    # For a in S, b,c in S with a<b<=c: need a not divide b+c
    # If all elements > n/2, then b+c > n, and for a > n/2, a divides b+c only if b+c is multiple of a
    # Safer: start with (2n/3, n] = (170.67, 256] => 171 to 256
    base_start = 171
    S = set(range(base_start, n + 1))
    
    def can_add(a, S):
        """Check if element a can be added to set S without violating property."""
        # For all b, c in S with a < b <= c: a must not divide (b+c)
        # Since a is being added and we're adding smaller elements, a < all elements in S
        # So we need: for all b, c in S with b <= c, a does not divide (b+c)
        elements = list(S)
        for i in range(len(elements)):
            b = elements[i]
            for j in range(i, len(elements)):
                c = elements[j]
                if b <= c and (b + c) % a == 0:
                    return False
        return True
    
    # Greedy phase: try to add elements from small to large
    for a in range(1, base_start):
        if time.time() > deadline - 10:
            break
        if can_add(a, S):
            S.add(a)
    
    # Local search: try swaps to improve
    best_S = S.copy()
    best_size = len(S)
    
    # Try removing one element and adding a smaller one
    for _ in range(500):  # limited iterations
        if time.time() > deadline - 5:
            break
        
        # Try random swap: remove one element, try to add a smaller one
        if not S:
            break
        
        candidates_to_remove = list(S)
        random.shuffle(candidates_to_remove)
        
        for remove_elem in candidates_to_remove[:5]:  # try up to 5 removals
            S_test = S.copy()
            S_test.discard(remove_elem)
            
            # Try to add smaller elements
            improved = False
            for a in range(1, remove_elem):
                if a not in S_test and can_add(a, S_test):
                    S_test.add(a)
                    if len(S_test) > best_size:
                        S = S_test.copy()
                        best_S = S.copy()
                        best_size = len(S)
                        improved = True
                        break
            
            if improved:
                break
    
    # Final greedy pass on remaining elements
    for a in range(1, n + 1):
        if time.time() > deadline - 2:
            break
        if a not in S and can_add(a, S):
            S.add(a)
    
    if len(S) > best_size:
        best_S = S.copy()
        best_size = len(S)
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
