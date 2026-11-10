"""Search for two 3-element sets A,B in [1,128] maximizing |A+B| with pairwise-coprime distinct sums."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _pairwise_coprime(vals):
    lst = list(vals)
    n = len(lst)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(lst[i], lst[j]) != 1:
                return False
    return True

def _score(A, B):
    s = {a + b for a in A for b in B}
    if len(s) < 2:
        return -1
    if not _pairwise_coprime(s):
        return -1
    return len(s)

def _best_known():
    # Known valid: [1,7,13] + [4,10,16] = {5,11,17,13,17,23,19,25,29}
    # Distinct: {5,11,17,13,23,19,25,29} -> check pairwise coprime
    # 5,11,17,13,23,19,25,29
    # gcd(5,25)=5 -> NOT coprime! So this seed might be invalid?
    # Let me recheck: A=[1,7,13], B=[4,10,16]
    # Sums: 1+4=5, 1+10=11, 1+16=17, 7+4=11, 7+10=17, 7+16=23, 13+4=17, 13+10=23, 13+16=29
    # Distinct: {5, 11, 17, 23, 29}
    # All primes, so pairwise coprime. Score = 5.
    # But the task says score 1.0000 for this, and score = value/seed.
    # So seed value = 5, and 5/5 = 1.0. Good.
    return 5

def construct():
    deadline = time.time() + 80.0  # safety margin under 90s
    
    best_score = 0
    best_AB = None
    
    # Start with known valid solution
    A0 = [1, 7, 13]
    B0 = [4, 10, 16]
    s0 = _score(A0, B0)
    if s0 > best_score:
        best_score = s0
        best_AB = (A0[:], B0[:])
    
    # The maximum possible distinct sums is 9 (3x3), but with duplicates it's less.
    # We want to maximize the number of distinct sums that are pairwise coprime.
    # Let's do a structured search.
    
    # Strategy: 
    # 1. Try to find sets where all 9 sums are distinct and pairwise coprime.
    # 2. If not possible, try 8, 7, etc.
    
    # Precompute primes up to 256 (max sum = 128+128=256)
    primes = [n for n in range(2, 257) if _is_prime(n)]
    
    # For pairwise coprime distinct sums, a good strategy is to use primes as targets.
    # Let's try a randomized search with bias toward prime sums.
    
    rng = random.Random(42)
    
    # Also try some structured approaches
    # Approach 1: Choose A and B such that a+b are all distinct primes
    # This is like a magic square of primes.
    
    # Let's do a hill-climbing / random restart search
    def _random_set():
        return sorted(rng.sample(range(1, 129), 3))
    
    def _neighbor(A, B):
        # Mutate one element in A or B
        if rng.random() < 0.5:
            idx = rng.randint(0, 2)
            A[idx] = rng.randint(1, 128)
            # Ensure distinct
            while len(set(A)) < 3:
                A[idx] = rng.randint(1, 128)
            A.sort()
            return A[:], B[:]
        else:
            idx = rng.randint(0, 2)
            B[idx] = rng.randint(1, 128)
            while len(set(B)) < 3:
                B[idx] = rng.randint(1, 128)
            B.sort()
            return A[:], B[:]
    
    # Simulated annealing / hill climbing
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Start from best or random
        if best_AB and rng.random() < 0.3:
            A_cur, B_cur = best_AB[0][:], best_AB[1][:]
        else:
            A_cur = _random_set()
            B_cur = _random_set()
        
        cur_score = _score(A_cur, B_cur)
        
        for step in range(200):
            if time.time() > deadline:
                break
            
            A_next, B_next = _neighbor(A_cur, B_cur)
            next_score = _score(A_next, B_next)
            
            if next_score > cur_score:
                A_cur, B_cur = A_next, B_next
                cur_score = next_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_AB = (A_cur[:], B_cur[:])
                    if best_score >= 9:
                        # Can't do better than 9
                        return [best_AB[0], best_AB[1]]
            elif next_score == cur_score:
                # Accept with some probability to escape local optima
                if rng.random() < 0.3:
                    A_cur, B_cur = A_next, B_next
    
    if best_AB:
        return [best_AB[0], best_AB[1]]
    else:
        return [A0, B0]

# EVOLVE-BLOCK-END
