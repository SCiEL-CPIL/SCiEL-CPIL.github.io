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
    """Search for optimal A, B using optimized difference-structure CSP."""
    deadline = time.time() + 75
    
    # Precompute GCD table
    max_s = 256
    gcd_table = [[0] * (max_s + 1) for _ in range(max_s + 1)]
    for i in range(2, max_s + 1):
        for j in range(i, max_s + 1):
            gcd_table[i][j] = gcd_table[j][i] = _gcd(i, j)
    
    def _check_pairwise_coprime_fast(sums_list):
        for i in range(len(sums_list)):
            for j in range(i + 1, len(sums_list)):
                if gcd_table[sums_list[i]][sums_list[j]] != 1:
                    return False
        return True

    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _eval(best_A, best_B)
    
    found_optimal = False
    max_diff = 30  # Reduced range for speed, increased coverage via filtering
    
    # Small primes for filtering
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    
    for da in range(1, min(max_diff, 127) + 1):
        if found_optimal or time.time() > deadline:
            break
        for db in range(1, min(max_diff, 127 - da + 1) + 1):
            if found_optimal or time.time() > deadline:
                break
            for dc in range(1, min(max_diff, 127) + 1):
                if found_optimal or time.time() > deadline:
                    break
                for dd in range(1, min(max_diff, 127 - dc + 1) + 1):
                    if found_optimal or time.time() > deadline:
                        break
                    
                    offsets = [
                        0, dc, dc + dd,
                        da, da + dc, da + dc + dd,
                        da + db, da + db + dc, da + db + dc + dd
                    ]
                    
                    max_offset = max(offsets)
                    min_base = 2
                    max_base = 256 - max_offset
                    
                    if min_base > max_base:
                        continue
                    
                    # Pre-compute bad residues for this pattern
                    # For each prime p, find residues r such that (base + r) % p == 0
                    # and there exists another offset s such that (base + s) % p == 0
                    # i.e., (r - s) % p == 0
                    # So if (offset_i - offset_j) % p == 0, then base cannot be -offset_i % p
                    bad_residues = {p: set() for p in small_primes}
                    
                    for p in small_primes:
                        for i in range(9):
                            for j in range(i + 1, 9):
                                if (offsets[i] - offsets[j]) % p == 0:
                                    # If offsets are congruent mod p, then base + offset_i and base + offset_j are congruent mod p.
                                    # If base + offset_i is 0 mod p, then both are 0 mod p -> GCD >= p
                                    # So base != -offset_i mod p
                                    bad_residues[p].add(-offsets[i] % p)
                    
                    b1_min_global = 1
                    b1_max_global = 128 - dc - dd
                    
                    if b1_min_global > b1_max_global:
                        continue
                    
                    for base in range(min_base, max_base + 1):
                        if found_optimal or time.time() > deadline:
                            break
                        
                        # Filter by small primes
                        valid = True
                        for p in small_primes:
                            if base % p in bad_residues[p]:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        b1_lo = max(b1_min_global, base + da + db - 128)
                        b1_hi = min(b1_max_global, base - 1)
                        
                        if b1_lo > b1_hi:
                            continue
                        
                        sums_9 = [base + off for off in offsets]
                        
                        if not _check_pairwise_coprime_fast(sums_9):
                            continue
                        
                        b1 = b1_lo
                        a1 = base - b1
                        a2 = a1 + da
                        a3 = a1 + da + db
                        b2 = b1 + dc
                        b3 = b1 + dc + dd
                        
                        A = [a1, a2, a3]
                        B = [b1, b2, b3]
                        
                        if not _is_valid(A, B):
                            continue
                        
                        score = _eval(A, B)
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                        
                        if best_score >= 9:
                            found_optimal = True
                            break
    
    if not found_optimal:
        rng = random.Random(42)
        for _ in range(50000):
            if time.time() > deadline:
                break
            A = list(rng.sample(range(1, 129), 3))
            B = list(rng.sample(range(1, 129), 3))
            score = _eval(A, B)
            if score > best_score:
                best_score = score
                best_A = A[:]
                best_B = B[:]
                if best_score >= 9:
                    return best_A, best_B

    return best_A, best_B

# Pre-compute the best answer
_best_result = _search()

def construct():
    return _best_result
# EVOLVE-BLOCK-END
