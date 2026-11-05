import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    # Primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    
    # We want to maximize L. Let's aim for L_max = 2000 initially, can expand if fast.
    # Given primes <= 1000, covering 1..1000 is the target.
    L_max = 2000
    
    # covered[i] is True if integer i+1 is covered
    covered = np.zeros(L_max + 1, dtype=bool)
    
    selected = []
    used_primes = set()
    budget = Fraction(1, 1)
    
    # Max-Gain Greedy:
    # At each step, find the (p, r) that covers the most currently uncovered numbers.
    # If multiple, prefer larger p (cheaper) to save budget.
    
    while time.time() < deadline:
        # Check if 1..L_max is fully covered
        if not np.any(~covered[1:L_max+1]):
            break
            
        best_p = -1
        best_r = -1
        best_gain = -1
        best_cost = Fraction(1, 1) # Lower cost is better if gain is same
        
        for p in primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if cost > budget:
                continue
                
            # Find the best residue r for this prime p
            max_gain_for_p = -1
            best_r_for_p = -1
            
            # Iterate through all possible residues
            # Optimization: We can just check all r in 0..p-1
            # This is O(p) per prime. Sum of p for primes <= 1000 is roughly 1000^2 / (2 log 1000) ~ 100k.
            # This is fast enough.
            
            for r in range(p):
                # Determine start index in covered array
                # Integer x is covered if x % p == r.
                # Index i corresponds to integer i+1.
                # So (i+1) % p == r => i % p == (r-1) % p.
                # If r == 0, i % p == p-1.
                # If r > 0, i % p == r-1.
                
                start_idx = (r - 1) % p
                # Ensure we start within bounds
                if start_idx >= L_max:
                    continue
                
                # Count uncovered numbers in this arithmetic progression
                # Slice: covered[start_idx : L_max : p]
                # We want to count Falses, which is ~covered
                gain = np.sum(~covered[start_idx : L_max : p])
                
                if gain > max_gain_for_p:
                    max_gain_for_p = gain
                    best_r_for_p = r
            
            if max_gain_for_p == -1:
                continue
                
            # Compare with global best
            # Primary: Maximize gain
            # Secondary: Minimize cost (Maximize p)
            if max_gain_for_p > best_gain or (max_gain_for_p == best_gain and p > best_p):
                best_gain = max_gain_for_p
                best_p = p
                best_r = best_r_for_p
                best_cost = cost
        
        if best_p == -1:
            break
            
        # Select (best_p, best_r)
        # Convert to Python int to satisfy evaluator
        selected.append([int(best_p), int(best_r)])
        used_primes.add(best_p)
        budget -= best_cost
        
        # Update covered array
        start_idx = (best_r - 1) % best_p
        covered[start_idx : L_max : best_p] = True
        
    # Determine L
    # Find the first uncovered integer in 1..L_max
    if not np.any(~covered[1:L_max+1]):
        L = L_max
    else:
        # Find first index i in 1..L_max where covered[i] is False
        # Note: covered[0] is integer 1. We want 1..L covered.
        # So we look for first False in covered[1:L_max+1]
        # If covered[0] is False, L=0? But we start covering from 1.
        # Let's check from index 0.
        if not covered[0]:
            L = 0
        else:
            # Find first False in covered[1:]
            # argmax returns first True in ~covered
            # We need to be careful with indices.
            # ~covered[1:L_max+1]
            # If all True, L = L_max
            # Else, index of first False in covered[1:L_max+1] is j.
            # The integer is j+1.
            # The largest L such that 1..L are covered is (j+1) - 1 = j.
            # Wait.
            # If covered[1] is False (integer 2), then 1 is covered, 2 is not. L=1.
            # Index 1 in array is integer 2.
            # first_false_idx = 1.
            # L should be 1.
            # So L = first_false_idx.
            
            # Let's verify:
            # If covered[0]=True, covered[1]=True, covered[2]=False.
            # Integers: 1(T), 2(T), 3(F).
            # L should be 2.
            # first_false_idx in covered[1:] would be 1 (index 2 in full array).
            # Wait, ~covered[1:L_max+1]
            # Index 0 of this slice is covered[1].
            # If covered[2] is False, slice index 1 is True in ~covered.
            # argmax returns 1.
            # L = 1? No, L should be 2.
            
            # Let's re-evaluate.
            # covered array: index 0 -> int 1, index 1 -> int 2, index 2 -> int 3.
            # If int 3 is uncovered, covered[2] is False.
            # We want L such that 1..L are covered.
            # 1, 2 are covered. 3 is not. L=2.
            # First uncovered index in full array is 2.
            # So L = first_uncovered_index.
            
            # Let's find first index i where covered[i] is False.
            # If covered[0] is False, L=0.
            # If covered[0] is True, covered[1] is False, L=1.
            # If covered[0] is True, covered[1] is True, covered[2] is False, L=2.
            
            # So L is simply the index of the first False in covered.
            
            # Check if any False in covered
            if not np.any(~covered):
                L = L_max
            else:
                first_false_idx = np.argmax(~covered)
                L = int(first_false_idx)
                
    return selected
