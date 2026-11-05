import time
import random
import math

def construct():
    start = time.time()
    DEADLINE = start + 75.0

    N = 12
    MAXV = 10000

    # 1. Precompute prime factors for all numbers up to 2*MAXV
    limit = 2 * MAXV
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    # 2. Define the set of allowed primes.
    # We want the sums of pairs to be "smooth" with respect to this set.
    # If we pick numbers that are multiples of small primes, their sums are likely to be as well.
    # Let's try a set of small primes.
    # If we use primes {2, 3, 5, 7, 11, 13, 17, 19, 23}, that's 9 primes.
    # Can we do better?
    # If we restrict numbers to be multiples of 2, sums are even (factor 2).
    # If we restrict numbers to be multiples of 3, sums are multiples of 3 (factor 3).
    # If we restrict numbers to be multiples of 6, sums are multiples of 6 (factors 2, 3).
    # Let's try to find a set of 12 numbers such that all sums have factors only from a small set P.
    # Strategy:
    # 1. Pick a candidate set of primes P.
    # 2. Generate a pool of numbers <= MAXV that are "P-smooth" (all prime factors in P).
    #    Wait, the numbers themselves don't need to be P-smooth, but their sums must be.
    #    Actually, if a number has a prime factor q not in P, it doesn't necessarily mean its sum with another number has q.
    #    Example: 5 + 7 = 12 = 2^2 * 3. 5 and 7 are not factors of 12.
    #    So we can't just filter the pool by smoothness of the numbers themselves.
    
    # Alternative Strategy:
    # Simulated Annealing on the set of 12 numbers.
    # Cost function: Number of distinct prime factors of all pairwise sums.
    # To make the search efficient, we can precompute the prime factors of all possible sums? No, sums go up to 20000.
    # Precomputing factors for 2..20000 is fast.
    
    # Let's stick to the SA approach but with a smarter initialization and move set.
    # The previous program did local search. SA allows escaping local minima.
    
    # Precompute factor sets for all sums 2..20000
    sum_factors = [set() for _ in range(limit + 1)]
    for s in range(2, limit + 1):
        sum_factors[s] = get_prime_factors(s)
        
    # We want to minimize the size of the union of these sets.
    
    # Initial solution:
    # Try to construct a solution where sums are highly composite or smooth.
    # A good heuristic: pick numbers that are multiples of a small number, say 2, 3, or 6.
    # If all numbers are multiples of 2, all sums are multiples of 2.
    # If all numbers are multiples of 3, all sums are multiples of 3.
    # If all numbers are multiples of 6, all sums are multiples of 6.
    # Let's try initializing with multiples of 2, 3, 5, etc. and then SA.
    
    def random_init():
        # Random distinct integers
        return random.sample(range(1, MAXV + 1), N)
    
    def structured_init(mult):
        # Multiples of mult
        candidates = [i * mult for i in range(1, MAXV // mult + 1)]
        if len(candidates) < N:
            return random_init()
        return random.sample(candidates, N)
        
    def cost(arr):
        union = set()
        for i in range(N):
            ai = arr[i]
            for j in range(i + 1, N):
                s = ai + arr[j]
                union.update(sum_factors[s])
        return len(union)
        
    def simulated_annealing(initial, t_start=1.0, t_end=0.001, alpha=0.999, max_iters=50000):
        cur = initial[:]
        cur_cost = cost(cur)
        best = cur[:]
        best_cost = cur_cost
        
        t = t_start
        for i in range(max_iters):
            if time.time() > DEADLINE:
                break
            if i % 100 == 0:
                t *= alpha
                if t < t_end:
                    break
                    
            # Move: Replace one element with a random one
            idx = random.randint(0, N - 1)
            new_val = random.randint(1, MAXV)
            if new_val in cur and new_val != cur[idx]:
                continue # Avoid duplicates if possible, though set check handles it
            
            new_arr = cur[:]
            new_arr[idx] = new_val
            
            # Check distinctness
            if len(set(new_arr)) != N:
                continue
                
            new_cost = cost(new_arr)
            delta = new_cost - cur_cost
            
            if delta <= 0 or random.random() < math.exp(-delta / t):
                cur = new_arr
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best = cur[:]
                    best_cost = cur_cost
                    
        return best, best_cost
        
    best_global = None
    best_global_cost = float('inf')
    
    # Try different structured starts
    starts = [
        structured_init(2),
        structured_init(3),
        structured_init(5),
        structured_init(6),
        structured_init(10),
        random_init(),
        random_init()
    ]
    
    for s in starts:
        if time.time() > DEADLINE:
            break
        res, res_cost = simulated_annealing(s)
        if res_cost < best_global_cost:
            best_global_cost = res_cost
            best_global = res
            
    # If we still have time, do more SA runs with different temperatures or moves
    # Or refine the best solution found with a stricter local search
    if best_global and time.time() < DEADLINE - 5:
        # Refine with local search (greedy improvement)
        cur = best_global[:]
        cur_cost = best_global_cost
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            for i in range(N):
                old_val = cur[i]
                # Try a few random replacements
                for _ in range(20):
                    new_val = random.randint(1, MAXV)
                    if new_val in cur and new_val != old_val:
                        continue
                    new_arr = cur[:]
                    new_arr[i] = new_val
                    if len(set(new_arr)) != N:
                        continue
                    nc = cost(new_arr)
                    if nc < cur_cost:
                        cur = new_arr
                        cur_cost = nc
                        improved = True
                        break # Restart loop for this i? Or just continue
                # Also try swapping with another element
                for j in range(i + 1, N):
                    new_arr = cur[:]
                    new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
                    if len(set(new_arr)) != N:
                        continue
                    nc = cost(new_arr)
                    if nc < cur_cost:
                        cur = new_arr
                        cur_cost = nc
                        improved = True
                        break
                if improved:
                    break
                    
        if cur_cost < best_global_cost:
            best_global_cost = cur_cost
            best_global = cur
            
    if best_global is None:
        best_global = random_init()
        
    return best_global
