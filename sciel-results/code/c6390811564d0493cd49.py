"""Rewritten program: local search over multiples of 6 with simulated annealing."""
import time
import random
from collections import Counter

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    n = 12
    max_k = 10000 // 6  # 1666
    
    # Precompute prime factorization for all possible k-sums (1..2*max_k)
    # k_i + k_j ranges from 2 to 2*max_k
    max_sum = 2 * max_k
    # sieve for smallest prime factor
    spf = list(range(max_sum + 1))
    for i in range(2, int(max_sum**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_sum + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_primes(x):
        """Get set of prime factors of x."""
        if x <= 1:
            return set()
        factors = set()
        while x > 1:
            p = spf[x]
            factors.add(p)
            while x % p == 0:
                x //= p
        return factors
    
    # Precompute prime sets for all possible k-sums
    prime_sets = [set() for _ in range(max_sum + 1)]
    for s in range(2, max_sum + 1):
        prime_sets[s] = get_primes(s)
    
    def count_primes(k_list):
        """Count distinct primes across all pairwise k-sums."""
        all_primes = set()
        for i in range(n):
            for j in range(i+1, n):
                s = k_list[i] + k_list[j]
                all_primes.update(prime_sets[s])
        return len(all_primes)
    
    # Initial solution: first n multiples of 6
    k = list(range(1, n+1))
    best_k = k[:]
    best_count = count_primes(k)
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    rng = random.Random(42)
    
    while time.time() < deadline and T > T_min:
        # Generate a move
        move_type = rng.random()
        new_k = k[:]
        
        if move_type < 0.5:
            # Swap two elements
            i, j = rng.sample(range(n), 2)
            new_k[i], new_k[j] = new_k[j], new_k[i]
        elif move_type < 0.8:
            # Replace one element with a random valid k
            i = rng.randint(0, n-1)
            new_val = rng.randint(1, max_k)
            # Ensure distinctness
            if new_val in new_k and new_val != new_k[i]:
                # Try a few times to find a unique value
                for _ in range(10):
                    new_val = rng.randint(1, max_k)
                    if new_val not in new_k or new_val == new_k[i]:
                        break
            new_k[i] = new_val
            # Check distinctness
            if len(set(new_k)) != n:
                continue
        else:
            # Perturb: change two elements
            i, j = rng.sample(range(n), 2)
            for idx in [i, j]:
                new_val = rng.randint(1, max_k)
                for _ in range(10):
                    if new_val not in new_k or new_val == new_k[idx]:
                        break
                    new_val = rng.randint(1, max_k)
                new_k[idx] = new_val
            if len(set(new_k)) != n:
                continue
        
        new_count = count_primes(new_k)
        delta = new_count - count_primes(k)
        
        if delta <= 0 or rng.random() < (2.718281828 ** (-delta / T)):
            k = new_k
            if new_count < best_count:
                best_count = new_count
                best_k = k[:]
        
        T *= cooling
    
    # Convert back to multiples of 6
    result = [6 * x for x in best_k]
    return result
