"""Rewritten program: Greedy with smallest affordable prime for prime residue covering."""
import sys
from fractions import Fraction
from math import isqrt

# Precompute primes up to 1000
def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

PRIMES_1000 = _primes_up_to(1000)
# Map prime to index for quick lookup
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES_1000)}

def construct():
    """
    Find a set of distinct primes p <= 1000 with residues r (0 <= r < p)
    such that sum(1/p) <= 1 and every integer 1..L is covered.
    Maximize L.
    """
    L_max = 100000
    covered = bytearray(L_max + 1)  # covered[i] = 1 if i is covered
    
    # Sort primes in ascending order (smallest first)
    primes_asc = sorted(PRIMES_1000)
    num_primes = len(primes_asc)
    is_used = [False] * num_primes
    
    solution = []
    remaining_budget = Fraction(1, 1)
    
    for n in range(1, L_max + 1):
        if covered[n]:
            continue
        
        # Find the smallest unused prime p such that 1/p <= remaining_budget
        # We want the smallest p (highest coverage density) that is affordable.
        chosen_idx = -1
        for idx in range(num_primes):
            if is_used[idx]:
                continue
            p = primes_asc[idx]
            if Fraction(1, p) <= remaining_budget:
                chosen_idx = idx
                break
        
        if chosen_idx == -1:
            # No affordable prime left
            break
        
        p = primes_asc[chosen_idx]
        is_used[chosen_idx] = True
        r = n % p
        solution.append([p, r])
        
        # Mark all m ≡ r (mod p) in 1..L_max as covered
        start = r if r > 0 else p
        if start < 1:
            start = 1
        for m in range(start, L_max + 1, p):
            covered[m] = 1
        
        remaining_budget -= Fraction(1, p)
        
        # If remaining budget is 0, we can't add more primes, but we continue checking coverage.

    # Compute the actual covered length L
    L = 0
    for i in range(1, L_max + 1):
        if covered[i]:
            L = i
        else:
            break
    
    return solution

# Test
if __name__ == "__main__":
    sol = construct()
    print(f"Number of classes: {len(sol)}")
    # Compute covered length
    L_max = 100000
    covered = bytearray(L_max + 1)
    for p, r in sol:
        start = r if r > 0 else p
        for m in range(start, L_max + 1, p):
            covered[m] = 1
    L = 0
    for i in range(1, L_max + 1):
        if covered[i]:
            L = i
        else:
            break
    print(f"Covered length L: {L}")
    # Check budget
    total = sum(Fraction(1, p) for p, r in sol)
    print(f"Total budget used: {total}")
