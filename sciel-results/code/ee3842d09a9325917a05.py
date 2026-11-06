import random
from itertools import combinations

def is_sidon(S):
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_4(A):
    for combo in combinations(A, 4):
        if is_sidon(combo):
            return True
    return False

def construct():
    # We want to find a set of 12 distinct integers in [0, 128]
    # such that no 4-element subset is a Sidon set.
    # This ensures the maximum Sidon subset size is 3.
    # (Any 4 distinct integers contain a non-AP triple, which is a Sidon set of size 3).
    
    best_set = list(range(12)) # Fallback: AP (Max Sidon 4)
    
    # Try to find a set with Max Sidon 3
    # Using random restarts and local search
    for restart in range(50):
        A = set()
        while len(A) < 12:
            A.add(random.randint(0, 128))
        current_A = list(A)
        
        # Local search: try to eliminate Sidon 4-subsets
        for step in range(1000):
            if not has_sidon_4(current_A):
                return current_A
            
            # Find a bad subset to guide the search?
            # Or just random swap.
            # Random swap is simpler and likely effective.
            i = random.randint(0, 11)
            old = current_A[i]
            new = random.randint(0, 128)
            if new == old:
                continue
            # Ensure distinctness
            if new in current_A:
                continue
            current_A[i] = new
            
        # If loop finishes without finding, keep best?
        # We don't track "best" easily without full score calc.
        # But if we find one, we return immediately.
        
    return best_set
