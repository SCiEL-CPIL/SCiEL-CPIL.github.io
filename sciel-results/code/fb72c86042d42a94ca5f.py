"""Program to find optimal A, B for pairwise-coprime sumset maximization."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations, product

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    vs = list(set(vals))
    n = len(vs)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _eval(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) <= 1:
        return len(sums)
    if _pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def _is_valid(A, B):
    """Check if A and B are valid: 3 distinct integers in [1,128]."""
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    return True

def _search():
    """Search for optimal A, B using structured approach."""
    deadline = time.time() + 80  # 80s budget, leaving 10s safety margin
    
    # Precompute a sieve for fast primality test
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Precompute smallest prime factor for fast factorization
    spf = list(range(257))
    for i in range(2, int(math.isqrt(256)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, 257, i):
                if spf[j] == j:
                    spf[j] = i
    
    def _prime_factors(n):
        """Return set of prime factors of n."""
        if n < 2:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def _fast_eval(A, B):
        """Faster evaluation using prime factor tracking."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) <= 1:
            return len(sums)
        # Check pairwise coprime using prime factors
        prime_set = set()
        for s in sums:
            pf = _prime_factors(s)
            if prime_set & pf:
                return 0
            prime_set |= pf
        return len(sums)
    
    # Start with a known good baseline
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _fast_eval(best_A, best_B)
    
    rng = random.Random(12345)
    
    # Phase 1: Systematic search with structured A values
    # For each A, find B values that make sums prime
    a_candidates = list(range(1, 61))
    all_A = list(combinations(a_candidates, 3))
    
    for idx_A, A in enumerate(all_A):
        if time.time() > deadline:
            break
        
        A = list(A)
        
        # Find b values where all three sums a_i + b are prime
        good_bs = []
        for b in range(1, 129):
            s1 = A[0] + b
            s2 = A[1] + b
            s3 = A[2] + b
            if 2 <= s1 <= 256 and 2 <= s2 <= 256 and 2 <= s3 <= 256:
                if is_prime[s1] and is_prime[s2] and is_prime[s3]:
                    good_bs.append(b)
        
        # Try combinations of 3 from good_bs
        if len(good_bs) >= 3:
            if len(good_bs) <= 25:
                b_combos = list(combinations(good_bs, 3))
            else:
                b_combos = []
                for _ in range(100):
                    b_combos.append(tuple(sorted(rng.sample(good_bs, 3))))
                b_combos = list(set(b_combos))
            
            for B_tuple in b_combos:
                if time.time() > deadline:
                    break
                B = list(B_tuple)
                if len(set(B)) != 3:
                    continue
                score = _fast_eval(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A[:]
                    best_B = B[:]
                    if best_score >= 9:
                        return best_A, best_B
        
        # Also try random B for this A
        for _ in range(30):
            if time.time() > deadline:
                break
            B = list(rng.sample(range(1, 129), 3))
            score = _fast_eval(A, B)
            if score > best_score:
                best_score = score
                best_A = A[:]
                best_B = B[:]
                if best_score >= 9:
                    return best_A, best_B
    
    # Phase 2: Broader random search
    for _ in range(300000):
        if time.time() > deadline:
            break
        
        A = list(rng.sample(range(1, 129), 3))
        B = list(rng.sample(range(1, 129), 3))
        score = _fast_eval(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                return best_A, best_B
    
    # Phase 3: Local search from best solution
    if best_score < 9:
        for _ in range(300000):
            if time.time() > deadline:
                break
            if rng.random() < 0.5:
                idx = rng.randint(0, 2)
                new_val = rng.randint(1, 128)
                if new_val not in best_A:
                    old = best_A[idx]
                    best_A[idx] = new_val
                    score = _fast_eval(best_A, best_B)
                    if score >= best_score:
                        best_score = score
                        if best_score >= 9:
                            return best_A, best_B
                    else:
                        best_A[idx] = old
            else:
                idx = rng.randint(0, 2)
                new_val = rng.randint(1, 128)
                if new_val not in best_B:
                    old = best_B[idx]
                    best_B[idx] = new_val
                    score = _fast_eval(best_A, best_B)
                    if score >= best_score:
                        best_score = score
                        if best_score >= 9:
                            return best_A, best_B
                    else:
                        best_B[idx] = old
    
    return best_A, best_B

# Pre-compute the best answer
_best_result = _search()

def construct():
    return _best_result
# EVOLVE-BLOCK-END
