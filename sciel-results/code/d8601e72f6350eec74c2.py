import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    def pairwise_coprime(vals):
        distinct = list(set(vals))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def score_pair(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if not pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    # Initial best from seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_pair(best_A, best_B)
    
    # Current state for search
    cur_A = best_A[:]
    cur_B = best_B[:]
    cur_score = best_score
    
    def random_valid_set_parity(parity):
        """Generate 3 distinct values in [1,128] with given parity (0=even, 1=odd)"""
        if parity == 0:
            candidates = list(range(2, 129, 2))  # even: 2,4,...,128
        else:
            candidates = list(range(1, 129, 2))   # odd: 1,3,...,127
        return sorted(random.sample(candidates, 3))
    
    # Key insight: For all sums to be pairwise coprime, no two sums can share factor 2.
    # So all sums must be the same parity. Since a+b parity = parity(a) XOR parity(b),
    # all a's must have same parity and all b's must have opposite parity.
    # So either A is all odd and B is all even, or A is all even and B is all odd.
    
    # Precompute primes up to 256 (max sum is 128+128=256)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(256)
    
    # Strategy 1: Targeted search for 9 distinct sums (all pairwise coprime)
    # Try both parity assignments
    for a_parity in [1, 0]:  # 1=odd, 0=even
        b_parity = 1 - a_parity
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            cur_A = random_valid_set_parity(a_parity)
            cur_B = random_valid_set_parity(b_parity)
            cur_score = score_pair(cur_A, cur_B)
            if cur_score > best_score:
                best_score = cur_score
                best_A = cur_A[:]
                best_B = cur_B[:]
            
            # Local improvement: try to fix problematic sums
            if cur_score > 0 and cur_score < 9:
                # Try to improve by changing one element at a time
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random element to change
                    if random.random() < 0.5:
                        idx = random.randint(0, 2)
                        old_val = cur_A[idx]
                        new_val = random.choice([v for v in range(1, 129) if (v % 2) == a_parity and v != old_val and v not in cur_A])
                        if not new_val:
                            continue
                        cur_A[idx] = new_val
                        cur_A.sort()
                        new_score = score_pair(cur_A, cur_B)
                        if new_score > cur_score:
                            cur_score = new_score
                            if cur_score > best_score:
                                best_score = cur_score
                                best_A = cur_A[:]
                                best_B = cur_B[:]
                        else:
                            cur_A[idx] = old_val
                            cur_A.sort()
                    else:
                        idx = random.randint(0, 2)
                        old_val = cur_B[idx]
                        new_val = random.choice([v for v in range(1, 129) if (v % 2) == b_parity and v != old_val and v not in cur_B])
                        if not new_val:
                            continue
                        cur_B[idx] = new_val
                        cur_B.sort()
                        new_score = score_pair(cur_A, cur_B)
                        if new_score > cur_score:
                            cur_score = new_score
                            if cur_score > best_score:
                                best_score = cur_score
                                best_A = cur_A[:]
                                best_B = cur_B[:]
                        else:
                            cur_B[idx] = old_val
                            cur_B.sort()
            
            if best_score == 9:
                break
        if best_score == 9:
            break
    
    # Strategy 2: Broader search if we haven't found 9
    if best_score < 9:
        def random_valid_set():
            return sorted(random.sample(range(1, 129), 3))
        
        # Local search from current best
        cur_A = best_A[:]
        cur_B = best_B[:]
        cur_score = best_score
        
        while time.time() < deadline:
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                if random.random() < 0.5:
                    idx = random.randint(0, 2)
                    old_val = cur_A[idx]
                    new_val = random.randint(1, 128)
                    if new_val in cur_A:
                        continue
                    cur_A[idx] = new_val
                    cur_A.sort()
                    new_score = score_pair(cur_A, cur_B)
                    if new_score > cur_score:
                        cur_score = new_score
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_A = cur_A[:]
                            best_B = cur_B[:]
                    else:
                        cur_A[idx] = old_val
                        cur_A.sort()
                else:
                    idx = random.randint(0, 2)
                    old_val = cur_B[idx]
                    new_val = random.randint(1, 128)
                    if new_val in cur_B:
                        continue
                    cur_B[idx] = new_val
                    cur_B.sort()
                    new_score = score_pair(cur_A, cur_B)
                    if new_score > cur_score:
                        cur_score = new_score
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_A = cur_A[:]
                            best_B = cur_B[:]
                    else:
                        cur_B[idx] = old_val
                        cur_B.sort()
            
            if not improved:
                cur_A = random_valid_set()
                cur_B = random_valid_set()
                cur_score = score_pair(cur_A, cur_B)
                if cur_score > best_score:
                    best_score = cur_score
                    best_A = cur_A[:]
                    best_B = cur_B[:]
    
    return [best_A, best_B]
