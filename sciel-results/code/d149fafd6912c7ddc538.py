# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import math

def _primes_upto(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def construct():
    """
    Construct a set of prime residue classes covering 1..L for the largest possible L,
    subject to sum(1/p) <= 1 and distinct primes p <= 1000.
    
    Strategy: NumPy-based Greedy Frontier Extension.
    1. Maintain a boolean numpy array `covered` of size CAP+1.
    2. Maintain current prefix length L.
    3. At each step, find the first uncovered integer N = L + 1.
    4. Try adding each unused prime p with residue r = N % p.
    5. For each candidate, compute the new covered array and the new prefix length.
    6. Select the candidate that maximizes the new prefix length.
    7. Commit the best candidate and repeat until no prime can extend the prefix or budget is exhausted.
    """
    CAP = 10**5
    primes = _primes_upto(1000)
    
    covered = np.zeros(CAP + 1, dtype=bool)
    covered[0] = True # 0 is not in 1..L, but helps with indexing if needed, though we start at 1
    
    used_primes = set()
    budget_used = Fraction(0, 1)
    classes = []
    
    L = 0
    
    # Precompute prime list for fast access
    prime_list = primes
    
    while L < CAP:
        N = L + 1
        if covered[N]:
            # This shouldn't happen if we always advance L to the first False
            L += 1
            continue
            
        best_p = None
        best_r = None
        best_new_L = L
        best_new_covered = None
        
        # Try all unused primes
        for p in prime_list:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if budget_used + cost > Fraction(1, 1):
                continue
            
            r = N % p
            
            # Create mask for this prime and residue
            # Indices in [1, CAP] such that n % p == r
            # If r == 0, indices are p, 2p, ...
            # If r > 0, indices are r, r+p, r+2p, ...
            # Note: np.arange(start, stop, step)
            if r == 0:
                start = p
            else:
                start = r
            
            if start > CAP:
                # No numbers in range covered
                continue
                
            mask_indices = np.arange(start, CAP + 1, p)
            
            # We need to OR this mask into a copy of covered to evaluate
            # To avoid creating a full mask array every time (which is 10^5 bools),
            # we can just check the extension directly?
            # No, we need the new covered state for the next steps if we commit.
            # But for evaluation, we just need the new L.
            
            # Efficient evaluation of new L:
            # The new covered set is `covered` OR `mask_indices`.
            # We know `covered[1..L]` are all True.
            # We know `covered[N]` is False.
            # `mask_indices` contains N (since N % p == r).
            # So `covered[N]` will become True.
            
            # We need to find the first index > L that is NOT covered by `covered` AND NOT in `mask_indices`.
            
            # Let's create a boolean mask for speed
            # This allocation is 10^5 bytes, which is fast.
            mask = np.zeros(CAP + 1, dtype=bool)
            mask[mask_indices] = True
            
            new_covered = covered | mask
            
            # Find new prefix
            # We know 1..L are covered.
            # We check from N onwards.
            # We can use np.where on new_covered[N:] to find the first False.
            # But new_covered is 10^5 size.
            
            # Optimization: 
            # We can just loop from N until we hit a False.
            # But doing this in Python is slow.
            # Let's use numpy.
            
            # Find the first False in new_covered starting from N
            # Invert: True means covered, False means not.
            # We want the first index i >= N such that new_covered[i] is False.
            
            # Slice from N to CAP
            tail = new_covered[N:]
            
            if not np.any(~tail):
                new_L = CAP
            else:
                # Find first False
                # ~tail is True where not covered
                first_false_rel = np.argmax(~tail)
                new_L = N + first_false_rel - 1 # Index is 0-based in tail, so N + idx. But we want the last covered.
                # If first_false_rel is 0, new_L = N - 1? No.
                # If tail[0] is False, new_L = N - 1. But N is covered by mask.
                # Wait, N is in mask_indices. So new_covered[N] is True.
                # So tail[0] is True.
                # So first_false_rel will be >= 1 if N is covered.
                
                # Let's be precise.
                # new_covered is 1..CAP.
                # We want largest L' such that 1..L' are True.
                # We know 1..L are True.
                # We check L+1 (which is N).
                # If N is True, check N+1, etc.
                
                # Using np.argmax on ~tail:
                # If all True, argmax is 0? No, if all False, argmax is 0.
                # If all True, ~tail is all False. argmax is 0.
                # We handled "not any(~tail)" -> all True.
                
                # So ~tail has at least one True.
                # argmax returns the first True index.
                # That index corresponds to the first UNCOVERED number in the tail.
                # Let idx be that index.
                # The uncovered number is N + idx.
                # The last covered number is N + idx - 1.
                new_L = N + first_false_rel - 1
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_p = p
                best_r = r
                best_new_covered = new_covered
        
        if best_p is None or best_new_L <= L:
            # No prime can extend the prefix
            break
        
        # Commit
        covered = best_new_covered
        L = best_new_L
        used_primes.add(best_p)
        budget_used += Fraction(1, best_p)
        classes.append([best_p, best_r])
        
    return classes
# EVOLVE-BLOCK-END
