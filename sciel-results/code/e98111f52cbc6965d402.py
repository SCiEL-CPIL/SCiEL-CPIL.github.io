<<<<<<< SEARCH
import time
import random
from math import gcd
from functools import reduce

# Deadline guard
_START_TIME = time.time()
DEADLINE = 85.0  # seconds, safety margin before 90s kill

# Primes for modular checking
# We need primes large enough to distinguish subset sums.
# If |S| = k, there are 2^k sums. We need P > 2^k to avoid wrap-around collisions 
# that look like real collisions. 
# However, even if P < 2^k, a collision mod P implies a collision in integers 
# only if the difference is a multiple of P. 
# Since the max sum is sum(1/s) < 1 + 1/2 + ... < 4, the integer values 
# L/s are bounded. 
# Actually, the modular check is: 
# Does there exist distinct subsets A, B such that sum_{a in A} 1/a == sum_{b in B} 1/b?
# This is equivalent to sum_{a in A} P/a - sum_{b in B} P/b == 0 mod P.
# Let w_i = P * inv(s_i) mod P.
# We check if any subset sum of w_i is 0 (excluding empty set) or if any two subset sums are equal.
# Checking if any subset sum is 0 mod P is sufficient to detect collisions 
# if we consider the difference set. 
# Specifically, if sum(A) == sum(B), then sum(A \ B) - sum(B \ A) = 0.
# So we just need to check if there is a non-empty subset of {w_i} with sum 0 mod P.
# Wait, this is for the difference. 
# Let's stick to checking for duplicate subset sums. It's safer.
# But checking for a zero-sum subset is equivalent to checking for collisions 
# if we allow coefficients -1, 0, 1.
# Yes, dissociation is exactly that no non-trivial combination of -1,0,1 sums to 0.
# So we can check: for a given set of weights w, is there a subset with sum 0 mod P?
# This is the "Subset Sum" problem. 
# For k=15, 2^15 checks.
# We can do this for multiple primes.

def _is_dissociated_fast(S, primes=(1000000007, 1000000009)):
    """
    Check if S is dissociated using modular arithmetic.
    Returns True if likely dissociated.
    """
    n = len(S)
    if n == 0:
        return True
    if n > 20:
        # For very large n, we can't do 2^n. 
        # But max n is 22. 2^22 is 4 million. 
        # 4 million * 2 primes = 8 million ops. Might be slow in Python.
        # We'll try up to 18 for fast checks, and use a slower heuristic for >18.
        # Actually, let's just try up to 18. If we find a set of size 18, we are done.
        # If not, we might miss size 19+.
        # Let's optimize the loop.
        pass

    for p in primes:
        # Check if p divides any element
        valid = True
        for s in S:
            if s % p == 0:
                valid = False
                break
        if not valid:
            continue
        
        # Compute modular weights: w_i = P * inv(s_i) mod P
        # Note: 1/s mod P is inv(s). We want to check sum(1/s) collisions.
        # Collision: sum_{a in A} 1/a = sum_{b in B} 1/b
        # => sum_{a in A} P/a - sum_{b in B} P/b = 0 mod P
        # Let w_i = P * inv(s_i) mod P.
        # We need to check if there exist disjoint subsets A, B (not both empty) 
        # such that sum_{a in A} w_a == sum_{b in B} w_b mod P.
        # This is equivalent to checking if there is a non-empty subset C 
        # such that sum_{c in C} w_c == 0 mod P, where C is the symmetric difference?
        # No. 
        # If sum(A) == sum(B), then sum(A \ B) - sum(B \ A) = 0.
        # Let D = A \ B and E = B \ A. D and E are disjoint.
        # sum_{d in D} w_d - sum_{e in E} w_e = 0 mod P.
        # This is a signed subset sum problem.
        # We can check this by generating all subset sums of w, and checking 
        # if any sum appears twice? No, that's for positive sums.
        # For signed sums, we can just check if any subset sum is 0?
        # No, because we have negative signs.
        # However, if we consider the set of all subset sums of w, 
        # a collision sum(A) == sum(B) means sum(A) - sum(B) = 0.
        # This difference is a sum of elements with coefficients -1, 0, 1.
        # So we need to check if there is a non-empty subset of indices I 
        # and signs s_i in {-1, 1} such that sum s_i w_i = 0 mod P.
        
        # Efficient way:
        # Generate all 2^n subset sums of w.
        # If any sum (excluding 0) appears, it doesn't mean collision.
        # Collision means sum(A) == sum(B).
        # This is equivalent to: does the set of subset sums contain a duplicate?
        # Yes! 
        # If sum(A) == sum(B) mod P, and A != B, then we have a collision.
        # So we just need to check if the 2^n subset sums are distinct mod P.
        
        if n > 18:
            # Too slow for 2 primes if n is large.
            # We'll just do one prime for n > 18.
            if n > 20:
                # Fallback: assume valid if it passed smaller checks? 
                # Or just skip?
                # Let's try to be safe. If n > 20, we are in deep search.
                # We'll do a single prime check.
                pass

        seen = set()
        seen.add(0)
        # Iterate over all 2^n subsets
        # Optimization: use iterative DP or simple loop
        sums = [0]
        for w in [pow(s, p - 2, p) for s in S]:
            # Add w to all existing sums
            new_sums = [(s + w) % p for s in sums]
            sums.extend(new_sums)
            # Check for duplicates
            # If len(sums) != len(set(sums)), collision found
            # But set(sums) is O(N log N) or O(N).
            # For n=18, len(sums) = 2^18 = 262k.
            # Checking duplicates at each step is expensive.
            # Better to generate all, then check.
            
        # Actually, generating all sums and then checking for duplicates is O(2^n).
        # Checking for duplicates is O(2^n).
        # Total O(2^n).
        
        # Let's do it properly.
        all_sums = [0]
        for s in S:
            w = pow(s, p - 2, p)
            current_len = len(all_sums)
            for i in range(current_len):
                all_sums.append((all_sums[i] + w) % p)
        
        # Check for duplicates
        if len(all_sums) != len(set(all_sums)):
            return False
            
    return True
=======
import time
import random
from math import gcd
from functools import reduce
import numpy as np

# Deadline guard
_START_TIME = time.time()
DEADLINE = 85.0  # seconds, safety margin before 90s kill

# Use two large primes for CRT-like collision detection
PRIME1 = 1000000007
PRIME2 = 1000000009

def _is_dissociated_fast(S):
    """
    Check if S is dissociated using modular arithmetic with NumPy vectorization.
    Returns True if likely dissociated.
    """
    n = len(S)
    if n == 0:
        return True
    if n > 22:
        return False

    # Compute modular weights for both primes
    # w_i = inv(s_i) mod p
    S_arr = np.array(S, dtype=np.int64)
    
    for p in (PRIME1, PRIME2):
        # Check if p divides any element
        if np.any(S_arr % p == 0):
            continue
        
        # Compute modular inverses
        weights = np.array([pow(s, p - 2, p) for s in S], dtype=np.int64)
        
        # Generate all 2^n subset sums using iterative doubling
        # Start with [0]
        all_sums = np.array([0], dtype=np.int64)
        
        for w in weights:
            # Add w to all existing sums, then concatenate
            new_sums = (all_sums + w) % p
            all_sums = np.concatenate([all_sums, new_sums])
        
        # Check for duplicates: if len(all_sums) != len(unique), collision
        # Use sorting for efficient uniqueness check
        sorted_sums = np.sort(all_sums)
        # If any adjacent elements are equal, there's a duplicate
        if n >= 1:
            if np.any(np.diff(sorted_sums) == 0):
                return False
    
    return True
>>>>>>> REPLACE

