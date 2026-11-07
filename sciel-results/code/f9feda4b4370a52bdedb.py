import bisect
from collections import defaultdict
import numpy as np

def construct():
    # Search space parameters
    N = 10_000_000
    
    # Efficient sigma sieve using NumPy
    # sigma[n] = sum of divisors of n
    sigma_arr = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma_arr[i::i] += i
        
    # Convert to list for faster indexing in Python loop if needed, 
    # but we can also work with numpy arrays for grouping?
    # Grouping by value is easier with lists.
    sigma_list = sigma_arr.tolist()
    
    # Group numbers by their sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma_list[n]].append(n)
        
    best_a = 3
    best_b = 2
    best_err = abs(3/2 - 1.5) # This is 0, so we need to skip exact match or handle it
    # The task says: "A raw value of exactly 0 cannot be scored and counts as invalid."
    # So we must find a pair where a/b != 1.5 exactly.
    # If a=3, b=2, sigma(3)=4, sigma(2)=3. Not equal.
    # We need sigma(a)=sigma(b).
    
    # Initialize best_err to infinity
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # Iterate through groups
    for S, L in groups.items():
        if len(L) < 2:
            continue
            
        # Sort the list to allow binary search
        L.sort()
        
        # For each b in L, find a in L such that a/b is close to 1.5
        # a = 1.5 * b
        for b in L:
            target = 1.5 * b
            # Find insertion point for target in L
            idx = bisect.bisect_left(L, target)
            
            candidates = []
            if idx < len(L):
                candidates.append(L[idx])
            if idx > 0:
                candidates.append(L[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                # Check exact ratio to avoid 0 error
                if 2 * a == 3 * b:
                    continue
                    
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
    return [best_a, best_b]
