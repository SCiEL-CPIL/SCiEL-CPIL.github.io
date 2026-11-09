import random
from itertools import combinations
from collections import defaultdict
import time

def construct():
    # Generate all 4-subsets of 1..16
    # Use bitmasks for efficiency
    # Bit i corresponds to element i+1
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
        
    n_sets = len(all_sets)
    
    # Precompute intersections? No, too much memory.
    # We will compute on the fly.
    
    best_family = []
    deadline = time.time() + 55.0 # 5s margin
    
    # Helper to check if C can be added to S
    def can_add(C, S):
        # S is a list of masks
        groups = defaultdict(list)
        for A in S:
            K = A & C
            groups[K].append(A)
        
        for K, list_A in groups.items():
            if len(list_A) >= 2:
                # Check pairs in list_A
                # If A & B == K, then A, B, C form a sunflower
                # Note: A & B == K is the condition.
                # We need to check if any pair has intersection exactly K.
                for i in range(len(list_A)):
                    for j in range(i+1, len(list_A)):
                        if (list_A[i] & list_A[j]) == K:
                            return False
        return True

    # Run multiple randomized greedy attempts
    # The order of adding sets matters.
    # We can also try different initial sets.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.shuffle(all_sets)
        S = []
        for C in all_sets:
            if time.time() > deadline:
                break
            if can_add(C, S):
                S.append(C)
        
        if len(S) > len(best_family):
            best_family = S[:]
            # If we found a very large one, we might want to keep going to see if we can do better
            # Or we can break if we hit a theoretical upper bound?
            # Upper bound for k=4, n=16 is not trivially known, but likely < 200.
            
    # Convert best_family masks back to sorted lists
    result = []
    for mask in best_family:
        s = []
        for i in range(1, 17):
            if mask & (1 << (i - 1)):
                s.append(i)
        result.append(s)
        
    return result
