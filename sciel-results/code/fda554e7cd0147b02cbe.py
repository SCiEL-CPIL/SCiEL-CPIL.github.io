import numpy as np
import time
import math
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 10**7
    
    # 1. Compute sigma(n) for all n <= N using vectorized sieve
    # Using numpy int64 to avoid overflow (sigma(n) can be large, but for n<=1e7 it fits in int64 easily)
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline:
        return [1, 1]

    # 2. Group indices by sigma value
    # We need to map sigma value -> list of indices n
    # Converting numpy array to list of lists efficiently is tricky.
    # We can sort indices by sigma value.
    
    # Get indices 1..N
    indices = np.arange(1, N + 1, dtype=np.int64)
    sigmas = sigma[1:N+1]
    
    # Sort indices based on their sigma values
    # np.lexsort is stable and fast
    # We want to group by sigma.
    # Let's create an array of (sigma, index) pairs? No, just sort indices by sigma.
    
    # Sorting 10^7 elements takes some time but is feasible.
    # Alternatively, we can iterate and use a defaultdict.
    # Iterating 10^7 times in Python is slow (~0.5-1s). Sorting with numpy is fast.
    
    # Let's try the defaultdict approach first as it's simpler to implement correctly for grouping.
    # If it's too slow, we switch to numpy sorting.
    
    groups = defaultdict(list)
    
    # To speed up the Python loop, we can extract the numpy arrays and iterate.
    # But 10^7 iterations in pure Python might take ~1 second.
    # Let's check time.
    
    # Optimization: Only keep groups with size >= 2?
    # We can do a second pass to filter, but building the dict is the main cost.
    
    for i in range(N):
        s_val = int(sigmas[i])
        n_val = int(indices[i])
        groups[s_val].append(n_val)
        
    if time.time() > deadline:
        # Fallback: return best found so far? No, we haven't found any.
        # We can return a known good pair if we had one, but we don't.
        # Let's try to salvage by returning [1, 1] which is invalid (sigma(1)=1, sigma(1)=1, err=|1-1.5|=0.5)
        # Wait, sigma(1)=1. If a=1, b=1, sigma(a)=sigma(b). But a and b must be distinct?
        # Task says "approximate a ratio...". Usually implies distinct?
        # "Return a list [a, b]". It doesn't explicitly say distinct, but "pair" usually implies distinct in this context?
        # "minimize |a/b - 3/2|". If a=b, ratio is 1. Error is 0.5.
        # If we find a pair with error < 0.5, it's better.
        # Let's assume distinct is required or at least preferred.
        # Actually, if a=b, sigma(a)=sigma(b) is trivial.
        # Is a=b allowed? "approximate a ratio...".
        # Most likely distinct integers.
        # If a=b is allowed, the answer is trivial? No, we need to minimize error.
        # If a=b, error is 0.5.
        # There are definitely pairs with error < 0.5.
        # So a=b is not the optimal solution.
        
        # If we run out of time, we need to return *something* valid.
        # Let's pick a random pair from a large group if available, or just [1,1].
        # [1,1] is valid integers, sigma(1)=1, sigma(1)=1.
        # Is [1,1] considered valid? "positive Python integers". Yes.
        # "Require sigma(a)=sigma(b)". Yes.
        # "Minimize |a/b-3/2|". 0.5.
        # It's a valid candidate, just not good.
        return [1, 1]

    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # 3. Find best pairs within each group
    for s_val, n_list in groups.items():
        if len(n_list) < 2:
            continue
            
        if time.time() > deadline:
            break
            
        # Sort the list of n's
        # n_list is already somewhat ordered? No, it was appended in order of n.
        # So n_list is already sorted by n.
        
        # For each b in n_list, find a in n_list such that a/b ~ 1.5
        # a ~ 1.5 * b
        # Since n_list is sorted, we can use bisect.
        
        # To avoid O(k^2) for large k, we use binary search.
        # However, k is likely small.
        
        for i in range(len(n_list)):
            b = n_list[i]
            target_a = 1.5 * b
            
            # Find position where target_a would be inserted
            # We want a > b?
            # If a < b, ratio < 1. Error |a/b - 1.5| > 0.5.
            # We are looking for min error.
            # If we find a pair with error < 0.5, it must have a/b in (1.0, 2.0).
            # So a and b are close in magnitude.
            # We can search the whole list.
            
            pos = bisect.bisect_left(n_list, int(target_a))
            
            # Check pos and pos-1
            candidates = []
            if pos < len(n_list):
                candidates.append(n_list[pos])
            if pos > 0:
                candidates.append(n_list[pos-1])
                
            # Also check immediate neighbors in case float rounding or large gaps?
            # Bisect finds the exact position. The closest integers to 1.5*b in the list
            # will be at pos or pos-1 (or pos+1 if pos is out of bounds? No, pos is insertion point).
            # Actually, if target_a is between n_list[pos-1] and n_list[pos],
            # the closest one is either n_list[pos-1] or n_list[pos].
            
            for a in candidates:
                if a == b:
                    continue
                    
                # Calculate error |a/b - 1.5|
                # err = abs(a/b - 1.5) = abs(2a - 3b) / (2b)
                # Use integer arithmetic for numerator to be exact, then float for division
                num = abs(2 * a - 3 * b)
                den = 2 * b
                err = num / den
                
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
    # Final check
    if best_a == best_b:
        # If no distinct pair found (unlikely), return [1,1]
        # But [1,1] has error 0.5.
        # If best_err is inf, it means no pairs found?
        # sigma(1)=1. Any other n with sigma(n)=1? No, sigma(n)>=1+1=2 for n>1?
        # No, sigma(p)=1+p for prime p.
        # sigma(1)=1.
        # So group for 1 only contains {1}.
        # So no pair with sigma=1.
        
        # What if no groups have size >= 2?
        # Impossible. sigma(2)=3, sigma(3)=4...
        # There are many collisions.
        
        pass
        
    return [int(best_a), int(best_b)]

if __name__ == "__main__":
    construct()
