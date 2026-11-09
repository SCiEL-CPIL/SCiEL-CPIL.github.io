import itertools
import math
import random
import time

def is_coprime(a, b):
    return math.gcd(a, b) == 1

def is_pairwise_coprime(lst):
    n = len(lst)
    for i in range(n):
        for j in range(i+1, n):
            if not is_coprime(lst[i], lst[j]):
                return False
    return True

def evaluate(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct_sums = list(sums)
    if is_pairwise_coprime(distinct_sums):
        return len(distinct_sums)
    return 0

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    best_score = 0
    best_A = None
    best_B = None
    
    # Seed
    A_seed = [1, 7, 13]
    B_seed = [4, 10, 16]
    score_seed = evaluate(A_seed, B_seed)
    if score_seed > best_score:
        best_score = score_seed
        best_A = A_seed
        best_B = B_seed

    start_time = time.time()
    timeout = 55
    
    # Precompute primes up to 256 (max sum is 128+128=256)
    primes = [p for p in range(2, 257) if is_prime(p)]
    prime_set = set(primes)
    
    current_A = list(best_A)
    current_B = list(best_B)
    current_score = best_score
    
    # Strategy 1: Try to find A and B such that all 9 sums are distinct primes
    # This gives maximum cardinality 9
    # We need a+b to be prime for all a in A, b in B
    # And all 9 sums must be distinct
    
    # Use random search with bias toward values that produce prime sums
    while time.time() - start_time < timeout:
        # Random restarts
        if random.random() < 0.05 or current_score == 0:
            current_A = random.sample(range(1, 129), 3)
            current_B = random.sample(range(1, 129), 3)
            current_score = evaluate(current_A, current_B)
            if current_score > best_score:
                best_score = current_score
                best_A = current_A
                best_B = current_B
                if best_score == 9:
                    return [best_A, best_B]
            continue

        improved = False
        
        # Try to improve by focusing on prime sums
        # For each position in A, try values that make more prime sums
        for i in range(3):
            # Generate candidate values biased toward those that produce prime sums with current B
            candidates = []
            for _ in range(50):
                new_a = random.randint(1, 128)
                if new_a in current_A:
                    continue
                # Count how many prime sums this would create
                prime_count = sum(1 for b in current_B if is_prime(new_a + b))
                candidates.append((prime_count, new_a))
            
            # Sort by prime count descending, try top candidates
            candidates.sort(reverse=True)
            for prime_count, new_a in candidates[:10]:
                test_A = current_A.copy()
                test_A[i] = new_a
                test_score = evaluate(test_A, current_B)
                if test_score > current_score:
                    current_A = test_A
                    current_score = test_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_A = current_A
                        best_B = current_B
                        if best_score == 9:
                            return [best_A, current_B]
                    break
        
        # Try mutating B
        if not improved:
            for i in range(3):
                candidates = []
                for _ in range(50):
                    new_b = random.randint(1, 128)
                    if new_b in current_B:
                        continue
                    prime_count = sum(1 for a in current_A if is_prime(a + new_b))
                    candidates.append((prime_count, new_b))
                
                candidates.sort(reverse=True)
                for prime_count, new_b in candidates[:10]:
                    test_B = current_B.copy()
                    test_B[i] = new_b
                    test_score = evaluate(current_A, test_B)
                    if test_score > current_score:
                        current_B = test_B
                        current_score = test_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_A = current_A
                            best_B = current_B
                            if best_score == 9:
                                return [best_A, current_B]
                        break
        
        # If stuck, try a more targeted approach: 
        # Pick 3 distinct primes for sums, then find A and B
        if current_score < 9 and random.random() < 0.1:
            # Try to construct from prime sums
            # We need a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3 all prime and distinct
            # This is like a 3x3 magic square of primes
            # Let's try: fix A, then find B such that all sums are prime
            for _ in range(100):
                test_A = random.sample(range(1, 129), 3)
                # For each b, we need a1+b, a2+b, a3+b all prime
                valid_bs = []
                for b in range(1, 129):
                    if all(is_prime(a + b) for a in test_A):
                        valid_bs.append(b)
                
                if len(valid_bs) >= 3:
                    # Try combinations of 3 valid b's
                    for combo in itertools.combinations(valid_bs, 3):
                        test_B = list(combo)
                        test_score = evaluate(test_A, test_B)
                        if test_score > current_score:
                            current_A = test_A
                            current_B = test_B
                            current_score = test_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_A = current_A
                                best_B = current_B
                                if best_score == 9:
                                    return [best_A, current_B]
                            break
                    if improved:
                        break

    return [best_A, best_B]
