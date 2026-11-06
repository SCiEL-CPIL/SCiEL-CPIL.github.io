import time
import random
from math import isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    MAX_VAL = 10000
    N = 12
    
    # Precompute smallest prime factor up to 2*MAX_VAL
    MAX_SUM = 2 * MAX_VAL
    spf = list(range(MAX_SUM + 1))
    for i in range(2, isqrt(MAX_SUM) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        if n == 1:
            return frozenset()
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return frozenset(factors)
    
    # Precompute prime factors for all possible sums
    pf_cache = [None] * (MAX_SUM + 1)
    for s in range(2, MAX_SUM + 1):
        pf_cache[s] = get_prime_factors(s)
        
    def evaluate(a):
        """Return the set of all prime factors of pairwise sums."""
        primes = set()
        for i in range(N):
            for j in range(i+1, N):
                primes.update(pf_cache[a[i] + a[j]])
        return primes

    best_primes = None
    best_set = None
    
    def update_best(a, primes):
        nonlocal best_primes, best_set
        if best_primes is None or len(primes) < len(best_primes):
            best_primes = primes
            best_set = a[:]

    # Strategy 1: Search for a set of 12 numbers where all sums are P-smooth
    # for small P.
    # If we find such a set, the number of distinct primes is |P|.
    
    # Let's try to find 12 distinct integers k_i such that k_i + k_j is 7-smooth.
    # 7-smooth numbers are of form 2^a 3^b 5^c 7^d.
    
    def is_smooth(n, P_set):
        if n <= 1:
            return True
        return pf_cache[n].issubset(P_set)
    
    # Generate smooth numbers up to 10000
    def get_smooth_numbers(P_set, limit):
        smooths = []
        for i in range(2, limit + 1):
            if is_smooth(i, P_set):
                smooths.append(i)
        return smooths

    # Try prime sets of size 3, 4, 5
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    found_smooth_solution = False
    
    # Try 3 primes first
    for i in range(len(small_primes)):
        for j in range(i+1, len(small_primes)):
            for k in range(j+1, len(small_primes)):
                P = frozenset([small_primes[i], small_primes[j], small_primes[k]])
                if time.time() > deadline - 10.0:
                    break
                
                # Find smooth numbers
                smooths = get_smooth_numbers(P, MAX_VAL)
                if len(smooths) < N:
                    continue
                
                # We need 12 distinct numbers from smooths such that their sums are also in smooths?
                # No, the sums just need to be P-smooth.
                # The numbers themselves don't need to be P-smooth, but it helps.
                # Actually, if a_i are P-smooth, a_i+a_j is not necessarily P-smooth.
                # Example: 2+3=5 (ok), 2+5=7 (ok if 7 in P), 3+5=8 (ok).
                # But 6+10=16 (ok).
                # Let's just search for a clique in the graph where nodes are [1..10000]
                # and edges connect u,v if u+v is P-smooth.
                
                # To make it faster, restrict nodes to P-smooth numbers?
                # If u,v are P-smooth, u+v might not be.
                # But if we find a clique of size 12 in the "sum is P-smooth" graph,
                # we are done.
                
                # Heuristic: Start with small P-smooth numbers.
                # Let's try to build a set greedily.
                
                # Precompute which pairs of smooth numbers sum to a smooth number?
                # Too expensive for all pairs.
                
                # Let's try local search on the set of P-smooth numbers.
                # Initial set: first N smooth numbers.
                cur_set = smooths[:N]
                
                # Check if valid
                def check_valid(s):
                    for u, v in combinations(s, 2):
                        if not is_smooth(u + v, P):
                            return False
                    return True
                
                if check_valid(cur_set):
                    update_best(cur_set, P)
                    found_smooth_solution = True
                    break
                
                # If not, try to fix it
                improved = True
                iterations = 0
                while improved and iterations < 50 and time.time() < deadline - 10.0:
                    improved = False
                    iterations += 1
                    for idx in range(N):
                        old_val = cur_set[idx]
                        # Try replacing with another smooth number
                        # Pick a random smooth number not in set
                        candidates = [s for s in smooths if s not in cur_set]
                        if not candidates:
                            continue
                        random.shuffle(candidates)
                        for new_val in candidates[:20]:
                            cur_set[idx] = new_val
                            if check_valid(cur_set):
                                update_best(cur_set, P)
                                found_smooth_solution = True
                                improved = False
                                break
                            cur_set[idx] = old_val
                        if found_smooth_solution:
                            break
                        if improved:
                            break
                    if found_smooth_solution:
                        break
                
                if found_smooth_solution:
                    break
            if found_smooth_solution:
                break
        if found_smooth_solution:
            break
    if found_smooth_solution:
        break

    if found_smooth_solution:
        return best_set

    # Strategy 2: Local search minimizing number of distinct primes
    # Start with a good initial guess: Multiples of a smooth number
    # e.g., 2520 * k? No, too few.
    # Let's use a pool of "smooth-ish" numbers.
    
    # Pool: Numbers that are products of small primes (2,3,5,7)
    pool = []
    for a in range(0, 15):
        for b in range(0, 15):
            for c in range(0, 15):
                for d in range(0, 15):
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if 1 <= val <= MAX_VAL:
                        pool.append(val)
    pool = list(set(pool))
    pool.sort()
    
    if len(pool) < N:
        # Fallback to all numbers
        pool = list(range(1, MAX_VAL + 1))
        
    # Initial random sample from pool
    cur_set = random.sample(pool, N)
    cur_primes = evaluate(cur_set)
    update_best(cur_set, cur_primes)
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline - 5.0:
        improved = False
        iterations += 1
        indices = list(range(N))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5.0:
                break
            old_val = cur_set[i]
            # Try replacing with another value from pool
            # Pick random candidates from pool
            candidates = random.sample(pool, min(50, len(pool)))
            for new_val in candidates:
                if new_val == old_val or new_val in cur_set:
                    continue
                cur_set[i] = new_val
                new_primes = evaluate(cur_set)
                if len(new_primes) < len(cur_primes):
                    cur_primes = new_primes
                    update_best(cur_set, cur_primes)
                    improved = True
                    break
                cur_set[i] = old_val
            if improved:
                break

    # Strategy 3: General random search if we haven't found a good solution
    if best_primes is None or len(best_primes) > 5:
        for restart in range(50):
            if time.time() > deadline - 2.0:
                break
            cur_set = random.sample(range(1, MAX_VAL + 1), N)
            cur_primes = evaluate(cur_set)
            update_best(cur_set, cur_primes)
            
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                i = random.randint(0, N-1)
                old_val = cur_set[i]
                for _ in range(20):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val in cur_set:
                        continue
                    cur_set[i] = new_val
                    new_primes = evaluate(cur_set)
                    if len(new_primes) < len(cur_primes):
                        cur_primes = new_primes
                        update_best(cur_set, cur_primes)
                        improved = True
                        break
                    cur_set[i] = old_val

    return best_set

