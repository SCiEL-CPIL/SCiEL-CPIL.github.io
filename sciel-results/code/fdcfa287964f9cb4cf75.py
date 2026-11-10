"""Search for A, B (each 3 distinct ints in [1,128]) maximizing |A+B| with pairwise-coprime distinct sums."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin under 90s wall-clock

    def distinct_sums(A, B):
        return {a + b for a in A for b in B}

    def is_pairwise_coprime(vals):
        lst = list(vals)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)

    # Known solution with score 7 (from seed: [[1,7,13],[4,10,16]] -> sums {5,11,17,11,17,23,17,23,29} = {5,11,17,23,29})
    # Wait, let me recompute: 1+4=5, 1+10=11, 1+16=17, 7+4=11, 7+10=17, 7+16=23, 13+4=17, 13+10=23, 13+16=29
    # Distinct: {5, 11, 17, 23, 29}. That's 5 distinct sums, all pairwise coprime? 
    # gcd(5,11)=1, gcd(5,17)=1, gcd(5,23)=1, gcd(5,29)=1, gcd(11,17)=1, gcd(11,23)=1, gcd(11,29)=1, gcd(17,23)=1, gcd(17,29)=1, gcd(23,29)=1. Yes.
    # Score is 5, not 7. The observed score of 1.8 suggests the seed score was ~1.0 and 1.8 is 1.8x that.
    # Let me just search for the best possible.

    best_score = 0
    best_pair = None

    # Start with the seed
    seed_A, seed_B = [1, 7, 13], [4, 10, 16]
    s = score_of(seed_A, seed_B)
    if s > best_score:
        best_score = s
        best_pair = (seed_A[:], seed_B[:])

    # Random search
    rng = random.Random(42)
    
    # Strategy 1: Random pairs
    while time.time() < deadline - 10:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        sc = score_of(A, B)
        if sc > best_score:
            best_score = sc
            best_pair = (A[:], B[:])
            if best_score >= 9:
                break

    # Strategy 2: Structured search - try to find prime-like sums
    # Sums should be small primes or numbers with few small prime factors
    # Let's try sums that are all primes
    primes = [p for p in range(2, 257) if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]
    
    # Try to find 3x3 grid where all distinct sums are from a set of pairwise coprime numbers
    # Use local search: start from best, mutate one element at a time
    if best_pair and time.time() < deadline - 5:
        cur_A, cur_B = best_pair
        cur_score = best_score
        no_improve = 0
        while time.time() < deadline - 5 and no_improve < 200:
            improved = False
            # Try mutating each element of A
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in cur_A:
                        continue
                    new_A = cur_A[:]
                    new_A[i] = new_val
                    sc = score_of(new_A, cur_B)
                    if sc > cur_score:
                        cur_A = new_A
                        cur_score = sc
                        improved = True
                        no_improve = 0
                        if cur_score >= 9:
                            break
                if cur_score >= 9:
                    break
            if cur_score >= 9:
                break
            # Try mutating each element of B
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in cur_B:
                        continue
                    new_B = cur_B[:]
                    new_B[i] = new_val
                    sc = score_of(cur_A, new_B)
                    if sc > cur_score:
                        cur_B = new_B
                        cur_score = sc
                        improved = True
                        no_improve = 0
                        if cur_score >= 9:
                            break
                if cur_score >= 9:
                    break
            if cur_score >= 9:
                break
            if not improved:
                no_improve += 1
                # Random restart from current best with small perturbation
                cur_A = [max(1, min(128, x + rng.randint(-10, 10))) for x in cur_A]
                # Ensure distinct
                seen = set()
                new_A = []
                for x in cur_A:
                    if x not in seen and 1 <= x <= 128:
                        seen.add(x)
                        new_A.append(x)
                while len(new_A) < 3:
                    x = rng.randint(1, 128)
                    if x not in seen:
                        seen.add(x)
                        new_A.append(x)
                cur_A = sorted(new_A)
                cur_B = [max(1, min(128, x + rng.randint(-10, 10))) for x in cur_B]
                seen = set()
                new_B = []
                for x in cur_B:
                    if x not in seen and 1 <= x <= 128:
                        seen.add(x)
                        new_B.append(x)
                while len(new_B) < 3:
                    x = rng.randint(1, 128)
                    if x not in seen:
                        seen.add(x)
                        new_B.append(x)
                cur_B = sorted(new_B)
                cur_score = score_of(cur_A, cur_B)
                if cur_score < 0:
                    cur_A, cur_B = best_pair
                    cur_score = best_score

        if cur_score > best_score:
            best_score = cur_score
            best_pair = (cur_A[:], cur_B[:])

    if best_pair is None:
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_pair[0], best_pair[1]]
# EVOLVE-BLOCK-END
