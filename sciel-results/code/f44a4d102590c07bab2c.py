"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_pairwise_coprime(values):
    """Check if all distinct values in the list are pairwise coprime."""
    distinct = list(set(values))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _score_pair(A, B):
    """Return number of distinct pairwise-coprime sums, or -1 if not valid."""
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    if not _is_pairwise_coprime(distinct):
        return -1
    return len(distinct)

def _score_pair_fast(A, B):
    """Fast score using incremental prime-factor conflict detection."""
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    n = len(distinct)
    if n <= 1:
        return n
    # Check pairwise coprimality with early exit
    for i in range(n):
        vi = distinct[i]
        for j in range(i + 1, n):
            if math.gcd(vi, distinct[j]) != 1:
                return -1
    return n

def construct():
    deadline = time.time() + 85  # 85s out of 90s budget
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Verify the seed is valid
    seed_sums = [a + b for a in [1, 7, 13] for b in [4, 10, 16]]
    seed_distinct = list(set(seed_sums))
    if _is_pairwise_coprime(seed_distinct):
        best_count = len(seed_distinct)
    
    rng = random.Random(42)
    
    # KEY INSIGHT: If A is all odd and B is all even (or vice versa),
    # then all sums a+b are odd, so factor 2 is automatically eliminated.
    # This dramatically increases the density of valid candidates.
    
    # Precompute odd and even numbers in [1, 128]
    odds = list(range(1, 129, 2))   # 64 odds
    evens = list(range(2, 129, 2))  # 64 evens
    
    # Strategy 1: Parity-mismatched random search (much denser valid candidates)
    # A from odds, B from evens (or A from evens, B from odds)
    for iteration in range(500000):
        if time.time() > deadline:
            break
        # Alternate between odd-A/even-B and even-A/odd-B
        if iteration % 2 == 0:
            A = sorted(rng.sample(odds, 3))
            B = sorted(rng.sample(evens, 3))
        else:
            A = sorted(rng.sample(evens, 3))
            B = sorted(rng.sample(odds, 3))
        s = _score_pair_fast(A, B)
        if s > best_count:
            best_count = s
            best_A = A[:]
            best_B = B[:]
            if best_count >= 9:
                break
    
    if best_count >= 9:
        return [best_A, best_B]
    
    # Strategy 2: Local search / hill climbing around best found
    if best_count >= 7:
        # Try perturbing individual elements
        for _ in range(200000):
            if time.time() > deadline:
                break
            # Pick a random element to perturb
            A2 = best_A[:]
            B2 = best_B[:]
            if rng.random() < 0.5:
                idx = rng.randint(0, 2)
                # Replace with nearby odd or even number
                val = A2[idx]
                parity = val % 2
                # Get numbers with same parity in range
                if parity == 1:
                    candidates = [x for x in odds if x != val]
                else:
                    candidates = [x for x in evens if x != val]
                if candidates:
                    new_val = rng.choice(candidates)
                    A2[idx] = new_val
                    A2 = sorted(A2)
                    # Ensure distinctness
                    if len(set(A2)) < 3:
                        continue
            else:
                idx = rng.randint(0, 2)
                val = B2[idx]
                parity = val % 2
                if parity == 1:
                    candidates = [x for x in odds if x != val]
                else:
                    candidates = [x for x in evens if x != val]
                if candidates:
                    new_val = rng.choice(candidates)
                    B2[idx] = new_val
                    B2 = sorted(B2)
                    if len(set(B2)) < 3:
                        continue
            s = _score_pair_fast(A2, B2)
            if s > best_count:
                best_count = s
                best_A = A2[:]
                best_B = B2[:]
                if best_count >= 9:
                    break
    
    if best_count >= 9:
        return [best_A, best_B]
    
    # Strategy 3: General random search (including same-parity sets)
    for _ in range(500000):
        if time.time() > deadline:
            break
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = _score_pair_fast(A, B)
        if s > best_count:
            best_count = s
            best_A = A[:]
            best_B = B[:]
            if best_count >= 9:
                break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
