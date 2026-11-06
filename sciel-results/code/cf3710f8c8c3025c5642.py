import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0 # Safety margin
    
    # Precompute Smallest Prime Factor (SPF) up to 20000
    max_val = 20000
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
        
    def get_prime_count(arr):
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                if s <= max_val:
                    primes.update(get_prime_factors(s))
                else:
                    # Fallback for large sums, though unlikely with b_i <= 1666
                    # But a_i <= 10000, so sum <= 20000.
                    # If we scale differently, we might exceed.
                    # With a_i = 6*b_i, max sum is 6*1666 + 6*1665 = 20000 approx.
                    # Actually max b_i is 1666, so max sum is 6*(1666+1665) = 6*3331 = 19986.
                    pass
        return len(primes), primes

    # Initial heuristic: Odd numbers scaled by 6
    # b_i in {1, 3, ..., 23}
    b = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    a = [6 * x for x in b]
    best_score, _ = get_prime_count(b)
    best_b = b[:]
    
    # Local Search
    # We search on b, but evaluate on b (since primes(6(b_i+b_j)) = {2,3} U primes(b_i+b_j))
    # So minimizing primes(b_i+b_j) is equivalent.
    
    def eval_b(arr_b):
        primes = set()
        n = len(arr_b)
        for i in range(n):
            for j in range(i+1, n):
                s = arr_b[i] + arr_b[j]
                primes.update(get_prime_factors(s))
        return len(primes), primes

    current_score, _ = eval_b(best_b)
    
    max_iters = 5000
    for _ in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Pick a random index to mutate
        k = random.randint(0, 11)
        old_val = best_b[k]
        
        # Try a few random new values
        improved = False
        for _ in range(5):
            # New value should be distinct from others
            # Range 1 to 1666
            new_val = random.randint(1, 1666)
            if new_val in best_b and new_val != old_val:
                continue
            if new_val == old_val:
                continue
                
            # Swap
            best_b[k] = new_val
            new_score, _ = eval_b(best_b)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                break
            elif new_score == current_score:
                # Tie-breaker: smaller sum of squares to keep numbers small?
                # Or just accept to escape plateaus?
                # Let's accept with small probability or if sum decreases
                old_sum = sum(x*x for x in best_b)
                best_b[k] = old_val
                # Calculate sum for new_val case
                # This is expensive. Let's just accept 10% of ties.
                if random.random() < 0.1:
                    best_b[k] = new_val
                    improved = True
                    break
                else:
                    best_b[k] = old_val
            else:
                best_b[k] = old_val
        
        if improved:
            # Restart counter if needed, but simple loop is fine
            pass

    # Final result
    a = [6 * x for x in best_b]
    # Ensure distinctness and bounds
    # b are distinct by construction (check needed?)
    # random.randint might pick same value if not careful, but we checked `if new_val in best_b`
    # Wait, `if new_val in best_b` checks if it's already there.
    # If we change k, old_val is removed.
    # So `if new_val in best_b` is correct for distinctness.
    
    return a
