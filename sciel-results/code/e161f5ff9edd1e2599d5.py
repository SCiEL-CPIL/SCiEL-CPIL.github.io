import math
import random
import time

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

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if _gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _validate(A, B):
    # A and B must be 3 distinct integers in [1,128]
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    for x in A:
        if x < 1 or x > 128:
            return False
    for x in B:
        if x < 1 or x > 128:
            return False
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if not _pairwise_coprime(list(sums)):
        return False
    return True

def _score(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    return len(sums)

def construct():
    deadline = time.time() + 80.0  # 80s limit, 10s safety margin
    
    # Precompute primes up to 256
    primes = [n for n in range(2, 257) if _is_prime(n)]
    prime_set = set(primes)
    
    best_score = 0
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    # Strategy 1: Try to find 9 distinct prime sums
    # For a given B, find all a in [1,128] such that a+b is prime for all b in B
    # Then pick 3 such a values that give 9 distinct sums
    
    # Generate candidate B sets (3-element subsets of [1,128])
    # To be efficient, focus on B with small elements
    
    # First, try structured search
    # For B = {b1,b2,b3}, valid a must satisfy: a+b1, a+b2, a+b3 all prime
    # This is equivalent to a being in intersection of (primes - b1), (primes - b2), (primes - b3)
    
    # Let's try random B and find valid A
    random.seed(42)
    
    # Precompute for each b, the set of a values in [1,128] such that a+b is prime
    # a+b is prime means a = prime - b, so a must be in {p - b : p in primes, 1 <= p-b <= 128}
    valid_a_for_b = {}
    for b in range(1, 129):
        valid_a_for_b[b] = set()
        for p in primes:
            a = p - b
            if 1 <= a <= 128:
                valid_a_for_b[b].add(a)
    
    # Now search: pick random B, find intersection of valid_a_for_b[b] for b in B
    # If intersection has >= 3 elements, pick 3 and check distinctness of sums
    
    tried_B = set()
    attempts = 0
    
    # Phase 1: Random search over B
    while time.time() < deadline and attempts < 200000:
        attempts += 1
        B = tuple(sorted(random.sample(range(1, 129), 3)))
        if B in tried_B:
            continue
        tried_B.add(B)
        
        # Find valid A values
        valid_asis = valid_a_for_b[B[0]] & valid_a_for_b[B[1]] & valid_a_for_b[B[2]]
        if len(valid_asis) < 3:
            continue
        
        # Try to pick 3 a values from valid_asis such that all 9 sums are distinct primes
        valid_asis_list = sorted(valid_asis)
        
        # Try a few combinations
        found = False
        for i in range(min(5, len(valid_asis_list))):
            for j in range(i+1, min(8, len(valid_asis_list))):
                for k in range(j+1, min(11, len(valid_asis_list))):
                    A = [valid_asis_list[i], valid_asis_list[j], valid_asis_list[k]]
                    sums = set()
                    all_prime = True
                    for a in A:
                        for b in B:
                            s = a + b
                            sums.add(s)
                            if s not in prime_set:
                                all_prime = False
                                break
                        if not all_prime:
                            break
                    if all_prime and len(sums) == 9:
                        # All 9 sums are distinct primes -> pairwise coprime
                        if _score(A, list(B)) > best_score:
                            best_score = 9
                            best_result = [A, list(B)]
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if best_score == 9:
            break
    
    # Phase 2: If we found 9, we're done. Otherwise, try to maximize with pairwise coprime (not necessarily prime)
    if best_score < 9:
        # Random search for pairwise coprime sums
        while time.time() < deadline and attempts < 500000:
            attempts += 1
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if _pairwise_coprime(list(sums)):
                s = len(sums)
                if s > best_score:
                    best_score = s
                    best_result = [A, B]
    
    # Phase 3: Local search around best result
    if best_score < 9 and time.time() < deadline:
        # Try perturbing the best result
        A = list(best_result[0])
        B = list(best_result[1])
        for _ in range(100000):
            if time.time() > deadline:
                break
            # Randomly change one element
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old = A[idx]
                A[idx] = random.randint(1, 128)
                if len(set(A)) != 3:
                    A[idx] = old
                    continue
            else:
                idx = random.randint(0, 2)
                old = B[idx]
                B[idx] = random.randint(1, 128)
                if len(set(B)) != 3:
                    B[idx] = old
                    continue
            
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if _pairwise_coprime(list(sums)):
                s = len(sums)
                if s > best_score:
                    best_score = s
                    best_result = [A[:], B[:]]
    
    return best_result
