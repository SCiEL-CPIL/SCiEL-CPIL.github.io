import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Precompute primes up to 256 (max sum is 128+128)
    max_sum = 256
    is_prime = [True] * (max_sum + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(max_sum)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_sum + 1, i):
                is_prime[j] = False
    
    # Lists of odds and evens in [1, 128]
    odds = [x for x in range(1, 129) if x % 2 == 1]
    evens = [x for x in range(2, 129) if x % 2 == 0]

    best_score = 0
    best_A = None
    best_B = None

    def update(A, B):
        nonlocal best_score, best_A, best_B
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        distinct_sums = list(sums)
        # Check pairwise coprime
        valid = True
        for i in range(len(distinct_sums)):
            for j in range(i+1, len(distinct_sums)):
                if math.gcd(distinct_sums[i], distinct_sums[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            score = len(distinct_sums)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
                if best_score == 9:
                    return True
        return False

    # Strategy: Find A (odds) and B (evens) such that all 9 sums are prime.
    # This guarantees pairwise coprime (distinct primes) and max cardinality 9.
    
    # Iterate over all combinations of 3 evens for B
    # For each B, find at least 3 distinct odds A such that A+B elements are prime.
    
    for B in combinations(evens, 3):
        if time.time() > deadline:
            break
        
        b1, b2, b3 = B
        valid_ods = []
        
        for a in odds:
            s1 = a + b1
            s2 = a + b2
            s3 = a + b3
            
            # All sums must be prime
            if is_prime[s1] and is_prime[s2] and is_prime[s3]:
                valid_ods.append(a)
                
                # We only need 3 distinct odds
                if len(valid_ods) == 3:
                    A = valid_ods
                    if update(A, B):
                        return [best_A, best_B]
                    break

    # If the direct prime search didn't find 9, try the symmetric case (A evens, B odds)
    for B in combinations(odds, 3):
        if time.time() > deadline:
            break
        
        b1, b2, b3 = B
        valid_evens = []
        
        for a in evens:
            s1 = a + b1
            s2 = a + b2
            s3 = a + b3
            
            if is_prime[s1] and is_prime[s2] and is_prime[s3]:
                valid_evens.append(a)
                
                if len(valid_evens) == 3:
                    A = valid_evens
                    if update(A, B):
                        return [best_A, best_B]
                    break

    # Fallback: Local search if no 9-sum solution found (unlikely)
    if best_score < 9:
        # Initialize with a random valid-ish guess or the best found
        if best_A is None:
            best_A = random.sample(odds, 3)
            best_B = random.sample(evens, 3)
            update(best_A, best_B)
            
        current_A = list(best_A)
        current_B = list(best_B)
        current_score = best_score
        
        # Simple local search
        while time.time() < deadline - 2.0:
            improved = False
            
            # Mutate A
            for i in range(3):
                for _ in range(5):
                    new_a = random.choice(odds)
                    if new_a in current_A:
                        continue
                    test_A = current_A.copy()
                    test_A[i] = new_a
                    if update(test_A, current_B):
                        current_A = test_A
                        current_score = best_score
                        improved = True
                        if best_score == 9:
                            return [best_A, best_B]
                        break
                
            # Mutate B
            if not improved:
                for i in range(3):
                    for _ in range(5):
                        new_b = random.choice(evens)
                        if new_b in current_B:
                            continue
                        test_B = current_B.copy()
                        test_B[i] = new_b
                        if update(current_A, test_B):
                            current_B = test_B
                            current_score = best_score
                            improved = True
                            if best_score == 9:
                                return [best_A, best_B]
                            break
                            
            if not improved:
                # Random restart if stuck
                if random.random() < 0.1:
                    current_A = random.sample(odds, 3)
                    current_B = random.sample(evens, 3)
                    update(current_A, current_B)

    return [best_A, best_B]
