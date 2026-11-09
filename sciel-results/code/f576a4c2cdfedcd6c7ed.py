"""Pairwise-coprime sumset maximizer: structured prime-based search with DFS."""
import time
from math import gcd
from itertools import combinations

# ---- precompute primes up to 256 ----
def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

PRIMES = _sieve(256)

def _pfactors(n):
    fs = set()
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            fs.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        fs.add(n)
    return fs

# prime-factor set for each possible sum 2..256
PF = {}
for s in range(2, 257):
    PF[s] = _pfactors(s)

def _pairwise_coprime(vals):
    seen = set()
    for v in vals:
        for p in PF[v]:
            if p in seen:
                return False
            seen.add(p)
    return True

def _valid_set(lst):
    if len(set(lst)) != 3:
        return False
    for x in lst:
        if x < 1 or x > 128:
            return False
    return True

def _sums(A, B):
    return [a + b for a in A for b in B]

def _distinct_count(A, B):
    return len(set(_sums(A, B)))

def _is_valid(A, B):
    if not _valid_set(A) or not _valid_set(B):
        return False
    return _pairwise_coprime(set(_sums(A, B)))

def construct():
    t0 = time.time()
    deadline = t0 + 55.0   # safety margin before 60s

    best = ([1, 7, 13], [4, 10, 16])
    best_score = 0
    if _is_valid(best[0], best[1]):
        best_score = _distinct_count(best[0], best[1])

    all_vals = list(range(1, 129))
    
    # Strategy: Focus on finding 9 distinct pairwise-coprime sums
    # Key: sums should be primes or prime powers
    # Generate candidate A sets with structure: include 1, and small primes
    
    # Phase 1: Structured search - A contains 1, and two small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    # Generate candidate A sets
    candidate_A_sets = []
    
    # Type 1: A = [1, p, q] where p, q are small primes
    for i in range(len(small_primes)):
        for j in range(i + 1, len(small_primes)):
            p, q = small_primes[i], small_primes[j]
            if p + q <= 128:
                candidate_A_sets.append([1, p, q])
    
    # Type 2: A = [p, q, r] all small primes
    for combo in combinations(small_primes[:20], 3):
        if max(combo) <= 128:
            candidate_A_sets.append(list(combo))
    
    # Type 3: A includes 1 and small composite numbers
    for x in range(2, 30):
        for y in range(x + 1, 30):
            if gcd(x, y) == 1:
                candidate_A_sets.append([1, x, y])
    
    # Deduplicate
    seen_A = set()
    unique_A_sets = []
    for A in candidate_A_sets:
        key = tuple(sorted(A))
        if key not in seen_A:
            seen_A.add(key)
            unique_A_sets.append(A)
    
    # For each candidate A, find best B using greedy + DFS
    for A in unique_A_sets:
        if time.time() > deadline:
            break
        if not _valid_set(A):
            continue
        
        # Find all valid b values
        cand_b = []
        for b in all_vals:
            sums = [a + b for a in A]
            if _pairwise_coprime(sums):
                cand_b.append(b)
        
        if len(cand_b) < 3:
            continue
        
        # Precompute sums for each b
        b_sums = {b: [a + b for a in A] for b in cand_b}
        
        # DFS to find best B of size 3
        best_for_A = 0
        best_B_for_A = None
        
        def dfs(chosen, sum_set, used_primes, depth):
            nonlocal best_for_A, best_B_for_A
            if depth == 3:
                if len(sum_set) > best_for_A:
                    best_for_A = len(sum_set)
                    best_B_for_A = list(chosen)
                return
            
            # Upper bound pruning
            if len(sum_set) + (3 - depth) * 3 <= best_for_A:
                return
            
            start = chosen[-1] + 1 if chosen else 0
            for i in range(start, len(cand_b)):
                b = cand_b[i]
                new_sums = b_sums[b]
                new_sum_set = set(sum_set)
                new_sum_set.update(new_sums)
                
                # Check pairwise coprime
                ok = True
                new_used = set(used_primes)
                for s in new_sums:
                    for p in PF[s]:
                        if p in new_used:
                            ok = False
                            break
                        new_used.add(p)
                    if not ok:
                        break
                if not ok:
                    continue
                
                chosen.append(b)
                dfs(chosen, new_sum_set, new_used, depth + 1)
                chosen.pop()
                
                if best_for_A == 9:
                    return
                if time.time() > deadline:
                    return
        
        dfs([], set(), set(), 0)
        
        if best_for_A > best_score:
            best_score = best_for_A
            best = (A[:], best_B_for_A if best_B_for_A else best[1])
            if best_score == 9:
                break
    
    # Phase 2: If not found 9, do broader random search
    if best_score < 9 and time.time() < deadline - 5:
        import random
        random.seed(42)
        
        for _ in range(50000):
            if time.time() > deadline:
                break
            
            strategy = random.randint(0, 3)
            if strategy == 0:
                A = random.sample(range(1, 40), 3)
                B = random.sample(range(1, 40), 3)
            elif strategy == 1:
                A = random.sample(range(1, 25), 3)
                B = random.sample(range(1, 60), 3)
            elif strategy == 2:
                A = [1] + random.sample(range(2, 50), 2)
                random.shuffle(A)
                B = random.sample(range(1, 50), 3)
            else:
                A = random.sample(range(1, 129), 3)
                B = random.sample(range(1, 129), 3)
            
            if _is_valid(A, B):
                s = _distinct_count(A, B)
                if s > best_score:
                    best_score = s
                    best = (A[:], B[:])
                    if best_score == 9:
                        break
    
    return [list(best[0]), list(best[1])]
