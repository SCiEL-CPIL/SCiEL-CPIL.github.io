import time
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N = 512
    # Candidates are 1 to 511
    candidates = list(range(1, N))
    
    # Weights
    weights = [1.0 / (N - a) for a in candidates]
    
    # Find primes < 512
    primes = []
    for i in range(2, N):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    # Map each candidate to indices of primes that divide it
    # We need to build the constraint matrix A (len(primes) x N-1)
    # A[p_idx, a_idx] = 1 if primes[p_idx] divides candidates[a_idx]
    
    # Since N is small, we can build this quickly
    num_primes = len(primes)
    num_candidates = len(candidates)
    
    # Try MILP first
    try:
        import scipy.optimize as opt
        import numpy as np
        
        # Check if milp exists
        if not hasattr(opt, 'milp'):
            raise NotImplementedError("scipy.optimize.milp not available")
            
        c = np.array([-w for w in weights]) # Minimize negative weights
        
        A = np.zeros((num_primes, num_candidates))
        for a_idx, a in enumerate(candidates):
            for p_idx, p in enumerate(primes):
                if a % p == 0:
                    A[p_idx, a_idx] = 1
        
        constraints = opt.LinearConstraint(A, lb=0, ub=1)
        integrality = np.ones(num_candidates)
        bounds = [(0, 1)] * num_candidates
        
        # Solve
        res = opt.milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            # res.x contains the solution
            # Filter for x > 0.5
            selected = [candidates[i] for i in range(num_candidates) if res.x[i] > 0.5]
            # Verify pairwise coprime (safety check)
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if gcd(selected[i], selected[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return selected
            else:
                # Fallback if MILP result is somehow invalid (shouldn't happen)
                pass
        else:
            raise RuntimeError("MILP failed: " + res.message)

    except Exception:
        pass
    
    # Fallback: Local Search Heuristic
    # 1. Greedy Initialization
    # The seed program does this: iterate 511 down to 1, add if coprime to all chosen.
    # This is equivalent to sorting by weight descending and adding if compatible.
    chosen = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    
    # 2. Local Search Improvement
    # Precompute prime factors for quick conflict check?
    # Or just use gcd. N is small.
    
    # Helper to check if x can be added to set S (excluding x itself if in S)
    def is_compatible(x, S):
        for y in S:
            if x == y:
                continue
            if gcd(x, y) != 1:
                return False
        return True

    # We want to swap a in S with b not in S if w(b) > w(a) and b is compatible with S\{a}
    
    # Map candidate to index in 'chosen' list for quick removal?
    # Or just use a set for 'chosen' and a list for order?
    # Let's use a set for 'chosen' to check membership O(1), and a list for iteration.
    # Actually, we need to calculate score.
    
    # Let's use a dictionary or set for current chosen numbers.
    chosen_set = set(chosen)
    
    # Precompute weights for all a
    w = {a: 1.0 / (512 - a) for a in range(1, 512)}
    
    # Sort candidates by weight descending for potential addition/swap
    # We only care about swaps that improve score.
    # So for each a in chosen, we look for the BEST b not in chosen that is compatible with chosen-{a}.
    # If w(b) > w(a), swap.
    
    improved = True
    while improved:
        improved = False
        # Iterate over a copy of the list to allow modification
        for a in list(chosen):
            # Temporarily remove a
            chosen_set.remove(a)
            
            best_b = None
            best_w = w[a] # Threshold
            
            # Look for b in 1..511 not in chosen_set
            # Optimization: We can sort candidates by weight and stop early if w(b) <= best_w
            # But we need to check compatibility.
            # Let's just iterate through all candidates not in chosen_set.
            # 511 iterations per 'a', 511 'a's -> ~260k gcd checks per pass.
            # This is fast in Python if not too many passes.
            
            # To speed up, sort candidates by weight once?
            # Let's just iterate 511..1 (highest weight first)
            for b in range(511, 0, -1):
                if b in chosen_set:
                    continue
                if w[b] <= best_w:
                    break # Since we iterate descending weight, no further b will be better
                if is_compatible(b, chosen_set):
                    best_b = b
                    best_w = w[b]
                    # We found the best b for this a. Break.
                    break
            
            if best_b is not None:
                # Perform swap
                chosen_set.add(best_b)
                # Update list: remove a, add best_b
                # We need to keep the list 'chosen' in sync with 'chosen_set'
                # For simplicity, we can just rebuild the list at the end of the pass,
                # or manage it carefully.
                # Let's just mark that we improved and rebuild list later?
                # No, we need 'chosen' for the next 'a' iteration?
                # If we swap a with b, the set changes.
                # It's safer to do one swap, then restart the loop or continue with updated set.
                # Let's restart the while loop to be safe and simple.
                # Or just update the list and continue.
                # Updating list:
                idx = chosen.index(a)
                chosen[idx] = best_b
                improved = True
                break # Break inner loop, restart outer while? 
                # Breaking inner for loop, then 'improved' is True.
                # The while loop will repeat.
                # This ensures we don't use stale 'chosen' state.
            else:
                # No swap found, restore a
                chosen_set.add(a)
                
        # If we finished the for loop without breaking (no swaps made), improved is False
        
    return chosen

# EVOLVE-BLOCK-END
