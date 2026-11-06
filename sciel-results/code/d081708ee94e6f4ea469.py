"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import itertools
import time

# EVOLVE-BLOCK-START
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(nums[i], nums[j]) != 1:
                return False
    return True

def _score_pair(A, B):
    """Return number of distinct pairwise-coprime sums, or 0 if not all coprime."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if not _pairwise_coprime(list(sums)):
        return 0
    return len(sums)

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    best_score = 0
    best_result = [[1, 7, 13], [4, 10, 16]]  # fallback
    
    import random
    random.seed(42)
    
    # Key insight: if all sums are odd, factor 2 is eliminated.
    # A all-odd + B all-even → all sums odd. Or A all-even + B all-odd → all sums odd.
    # This dramatically increases the density of valid configurations.
    
    odd_pool = list(range(1, 129, 2))   # 64 odd numbers
    even_pool = list(range(2, 129, 2))  # 63 even numbers
    
    # Phase 1: Backtracking search - fix A, find B via constraint propagation
    # Use smaller pools for speed
    odd_small = odd_pool[:40]   # first 40 odd numbers: 1,3,5,...,79
    even_small = even_pool[:40] # first 40 even numbers: 2,4,6,...,80
    
    def find_B_for_A(A, B_pool, depth=0):
        """Backtrack to find B of size 3 from B_pool such that all 9 sums are pairwise coprime."""
        if len(A) != 3:
            return None
        if depth == 3:
            return A  # found valid B
        remaining = deadline - time.time()
        if remaining < 0.1:
            return None
        
        for i, b in enumerate(B_pool):
            if b in A:
                continue
            # Check: for each a in original_A, s = a+b must be coprime to all existing sums
            valid = True
            new_sums = []
            for a in orig_A:
                s = a + b
                for existing in used_sums:
                    if _gcd(s, existing) != 1:
                        valid = False
                        break
                if not valid:
                    break
                new_sums.append(s)
            if valid:
                A.append(b)
                used_sums.update(new_sums)
                result = find_B_for_A(A, B_pool[i+1:], depth+1)
                if result is not None:
                    return result
                used_sums.difference_update(new_sums)
                A.pop()
        return None
    
    # Try structured A from odd_small
    for A_tuple in itertools.combinations(odd_small, 3):
        if time.time() > deadline:
            break
        orig_A = list(A_tuple)
        used_sums = set()
        B_found = find_B_for_A([], even_small, 0)
        if B_found is not None:
            s = _score_pair(orig_A, B_found)
            if s > best_score:
                best_score = s
                best_result = [orig_A, B_found]
                if best_score == 9:
                    return best_result
    
    # Phase 2: Also try A from even_small, B from odd_small
    for A_tuple in itertools.combinations(even_small, 3):
        if time.time() > deadline:
            break
        orig_A = list(A_tuple)
        used_sums = set()
        B_found = find_B_for_A([], odd_small, 0)
        if B_found is not None:
            s = _score_pair(orig_A, B_found)
            if s > best_score:
                best_score = s
                best_result = [orig_A, B_found]
                if best_score == 9:
                    return best_result
    
    # Phase 3: Random search with parity bias
    for _ in range(500000):
        if time.time() > deadline:
            break
        if random.random() < 0.5:
            A = random.sample(odd_pool, 3)
            B = random.sample(even_pool, 3)
        else:
            A = random.sample(even_pool, 3)
            B = random.sample(odd_pool, 3)
        s = _score_pair(A, B)
        if s > best_score:
            best_score = s
            best_result = [A, B]
            if best_score == 9:
                return best_result
    
    # Phase 4: Local search around best
    if best_score >= 5:
        A0 = list(best_result[0])
        B0 = list(best_result[1])
        for _ in range(300000):
            if time.time() > deadline:
                break
            A = list(A0)
            B = list(B0)
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                A[idx] = random.choice(odd_pool if A[idx] % 2 == 1 else even_pool)
            else:
                idx = random.randint(0, 2)
                B[idx] = random.choice(odd_pool if B[idx] % 2 == 1 else even_pool)
            if len(set(A)) != 3 or len(set(B)) != 3:
                continue
            s = _score_pair(A, B)
            if s > best_score:
                best_score = s
                best_result = [A, B]
                A0 = A
                B0 = B
                if best_score == 9:
                    return best_result
    
    return best_result
# EVOLVE-BLOCK-END
