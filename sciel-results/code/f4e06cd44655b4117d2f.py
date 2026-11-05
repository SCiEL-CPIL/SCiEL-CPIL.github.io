# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import time
    import random
    import math
    from itertools import combinations

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def get_sums(A, B):
        return {a + b for a in A for b in B}

    def is_pairwise_coprime(sums):
        sums_list = list(sums)
        for i in range(len(sums_list)):
            for j in range(i + 1, len(sums_list)):
                if math.gcd(sums_list[i], sums_list[j]) != 1:
                    return False
        return True

    def score(A, B):
        if len(A) != 3 or len(B) != 3:
            return 0
        if len(set(A)) != 3 or len(set(B)) != 3:
            return 0
        for x in A:
            if not (1 <= x <= 128):
                return 0
        for x in B:
            if not (1 <= x <= 128):
                return 0
        sums = get_sums(A, B)
        if not is_pairwise_coprime(sums):
            return 0
        return len(sums)

    start_time = time.time()
    deadline = start_time + 80
    
    # Precompute primes up to 256
    primes = [i for i in range(2, 257) if is_prime(i)]
    is_prime_arr = [False] * 257
    for p in primes:
        is_prime_arr[p] = True

    evens = [x for x in range(1, 129) if x % 2 == 0]
    odds = [x for x in range(1, 129) if x % 2 == 1]
    
    # Precompute prime masks for odds
    # mask[a] has bit b set if a+b is prime
    prime_masks = {}
    for a in odds:
        mask = 0
        for b in evens:
            if a + b <= 256 and is_prime_arr[a + b]:
                mask |= (1 << b)
        prime_masks[a] = mask

    # Iterate over B combinations
    # We want to find A such that A+B are 9 distinct primes.
    # This requires A to be odd and B to be even (or vice versa).
    # We fix B as even.
    
    # To speed up, we can also precompute masks for B combinations?
    # No, just compute on the fly.
    
    search_start = time.time()
    
    # Optimization: Sort evens to maybe find small solutions first?
    # Already sorted.
    
    for b1_idx in range(len(evens)):
        if time.time() - search_start > 30:
            break
        b1 = evens[b1_idx]
        for b2_idx in range(b1_idx + 1, len(evens)):
            b2 = evens[b2_idx]
            # Early break if b1+b2 is too large? No, sums can be up to 256.
            # But if b1 and b2 are large, a+b will be large.
            # Primes are dense enough.
            
            mask_b12 = (1 << b1) | (1 << b2)
            
            for b3_idx in range(b2_idx + 1, len(evens)):
                b3 = evens[b3_idx]
                
                mask_b = mask_b12 | (1 << b3)
                
                # Find valid A's
                valid_as = []
                for a in odds:
                    if prime_masks[a] & mask_b == mask_b:
                        valid_as.append(a)
                
                if len(valid_as) < 3:
                    continue
                
                # Check triples
                if len(valid_as) <= 10:
                    for a1_idx in range(len(valid_as)):
                        a1 = valid_as[a1_idx]
                        for a2_idx in range(a1_idx + 1, len(valid_as)):
                            a2 = valid_as[a2_idx]
                            for a3_idx in range(a2_idx + 1, len(valid_as)):
                                a3 = valid_as[a3_idx]
                                A = [a1, a2, a3]
                                B = [b1, b2, b3]
                                sums = get_sums(A, B)
                                if len(sums) == 9 and is_pairwise_coprime(sums):
                                    return [A, B]
                else:
                    # Random sample
                    for _ in range(50):
                        A = random.sample(valid_as, 3)
                        B = [b1, b2, b3]
                        sums = get_sums(A, B)
                        if len(sums) == 9 and is_pairwise_coprime(sums):
                            return [A, B]
    
    # Fallback: Random search
    random.seed(42)
    best_A = [1, 2, 3]
    best_B = [4, 5, 6]
    best_score = score(best_A, best_B)
    
    A = best_A[:]
    B = best_B[:]
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        for _ in range(200):
            if time.time() > deadline: break
            side = random.choice(['A', 'B'])
            idx = random.randint(0, 2)
            new_val = random.randint(1, 128)
            
            if side == 'A':
                if new_val in A: continue
                old = A[idx]
                A[idx] = new_val
            else:
                if new_val in B: continue
                old = B[idx]
                B[idx] = new_val
                
            new_score = score(A, B)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_A = A[:]
                    best_B = B[:]
                    if best_score == 9:
                        return [best_A, best_B]
            else:
                if side == 'A':
                    A[idx] = old
                else:
                    B[idx] = old
                    
            if current_score == 9:
                return [A, B]
                
        if not improved:
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            current_score = score(A, B)
            if current_score > best_score:
                best_score = current_score
                best_A = A[:]
                best_B = B[:]
                
    return [best_A, best_B]
# EVOLVE-BLOCK-END
