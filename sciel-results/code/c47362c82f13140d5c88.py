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
    
    # Get all primes up to MAX_SUM
    primes_up_to = [i for i in range(2, MAX_SUM + 1) if spf[i] == i]
    
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
    
    # === NEW APPROACH: Targeted prime set construction ===
    # Strategy: Try small prime sets P. For each P, find 12 numbers where
    # all pairwise sums have prime factors only in P.
    
    # Candidate prime sets (small primes first)
    small_primes = primes_up_to[:30]  # primes up to ~113
    
    # Try prime sets of size 3, 4, 5, ...
    # For each set, try to construct numbers using a base multiplier approach
    
    # Precompute: for a given prime set P, which sums (2..20000) are "P-smooth"
    # (all prime factors in P)?
    
    def is_P_smooth(n, P_set):
        """Check if all prime factors of n are in P_set."""
        if n <= 1:
            return True
        return pf_cache[n].issubset(P_set)
    
    # Build P-smooth sum lookup for each candidate P
    # This is expensive, so we'll be selective about which P to try
    
    # Start with the most promising small prime sets
    # The idea: if P = {2, 3, 5}, then sums must be 5-smooth numbers
    # If P = {2, 3, 5, 7}, sums must be 7-smooth, etc.
    
    # For a prime set P, we want to find 12 distinct numbers x_1..x_12 in [1, 10000]
    # such that x_i + x_j is P-smooth for all i != j.
    
    # Approach: Use a base B that is P-smooth. Then x_i = B * k_i.
    # x_i + x_j = B * (k_i + k_j). This is P-smooth iff (k_i + k_j) is P-smooth
    # (since B is already P-smooth, the product is P-smooth iff k_i+k_j is P-smooth).
    
    # So we need k_i + k_j to be P-smooth for all pairs.
    
    # Let's try this for various P and B.
    
    # Precompute P-smooth numbers up to some limit for various P
    def build_P_smooth_list(P_set, limit):
        """Return sorted list of P-smooth numbers in [2, limit]."""
        result = []
        for n in range(2, limit + 1):
            if is_P_smooth(n, P_set):
                result.append(n)
        return result
    
    # Try prime sets in order of promise
    # Size 3 sets first
    prime_sets_to_try = []
    
    # All 3-prime sets from the first 15 primes
    for i in range(15):
        for j in range(i+1, 15):
            for k in range(j+1, 15):
                prime_sets_to_try.append(frozenset([small_primes[i], small_primes[j], small_primes[k]]))
    
    # Sort by product (smaller product = more P-smooth numbers = easier to find solutions)
    prime_sets_to_try.sort(key=lambda P: min(P) * max(P))
    
    # Also add some 4-prime sets
    for i in range(10):
        for j in range(i+1, 10):
            for k in range(j+1, 10):
                for l in range(k+1, 10):
                    prime_sets_to_try.append(frozenset([small_primes[i], small_primes[j], small_primes[k], small_primes[l]]))
    
    # Limit the number of sets to try based on time
    max_sets_to_try = 50
    
    # For each prime set, try to construct a solution
    for P_set in prime_sets_to_try[:max_sets_to_try]:
        if time.time() > deadline - 5.0:
            break
        
        # For this P_set, try various base multipliers B that are P-smooth
        # B should be large enough that B*k <= 10000 for reasonable k
        # and small enough that we have enough k values
        
        # Find P-smooth numbers that could serve as bases
        # B should be P-smooth and B * 20 <= 10000 (so k up to 20)
        # Actually, let's be more flexible
        
        # Try B = 1, and also try B being small P-smooth numbers
        candidate_bases = [1]
        # Add small P-smooth numbers
        for b in range(2, 100):
            if is_P_smooth(b, P_set):
                candidate_bases.append(b)
        
        found_solution = False
        
        for B in candidate_bases:
            if time.time() > deadline - 5.0:
                break
            
            # Max k value: B * k <= 10000, so k <= 10000 // B
            max_k = MAX_VAL // B
            if max_k < N:
                continue
            
            # We need k_i + k_j to be P-smooth for all pairs
            # k_i are distinct integers in [1, max_k]
            
            # Build list of P-smooth numbers up to 2*max_k (since k_i + k_j <= 2*max_k)
            smooth_sums = build_P_smooth_list(P_set, 2 * max_k)
            smooth_set = set(smooth_sums)
            
            # Now we need to find N distinct k values from [1, max_k]
            # such that all pairwise sums are in smooth_set
            
            # Use backtracking / constraint propagation
            # Start with greedy: try to build a set of k values
            
            # Initial attempt: k = 1, 2, ..., N
            cur_k = list(range(1, N + 1))
            
            # Check if this works
            def check_k_valid(k_list):
                for i in range(N):
                    for j in range(i+1, N):
                        if k_list[i] + k_list[j] not in smooth_set:
                            return False
                return True
            
            if check_k_valid(cur_k):
                # Found a solution!
                x_vals = [B * k for k in cur_k]
                if len(set(x_vals)) == N and all(1 <= x <= MAX_VAL for x in x_vals):
                    primes_used = set()
                    for x in x_vals:
                        primes_used.update(pf_cache[x])  # Actually we need sums
                    # Compute actual prime factors of sums
                    actual_primes = set()
                    for i in range(N):
                        for j in range(i+1, N):
                            actual_primes.update(pf_cache[x_vals[i] + x_vals[j]])
                    update_best(x_vals, actual_primes)
                    found_solution = True
                    break
            
            # If initial doesn't work, try local search
            improved = True
            iterations = 0
            while improved and iterations < 100 and time.time() < deadline - 5.0:
                improved = False
                iterations += 1
                indices = list(range(N))
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() > deadline - 5.0:
                        break
                    
                    old_k = cur_k[i]
                    # Try replacing with other values
                    for new_k in range(1, max_k + 1):
                        if new_k == old_k or new_k in cur_k:
                            continue
                        
                        cur_k[i] = new_k
                        if check_k_valid(cur_k):
                            # Found a valid configuration
                            x_vals = [B * k for k in cur_k]
                            if len(set(x_vals)) == N and all(1 <= x <= MAX_VAL for x in x_vals):
                                actual_primes = set()
                                for ii in range(N):
                                    for jj in range(ii+1, N):
                                        actual_primes.update(pf_cache[x_vals[ii] + x_vals[jj]])
                                update_best(x_vals, actual_primes)
                                found_solution = True
                                break
                        cur_k[i] = old_k
                    
                    if found_solution:
                        break
            
            if found_solution:
                break
        
        if found_solution and best_primes is not None and len(best_primes) <= 3:
            break
    
    # === FALLBACK: Original local search approach if targeted construction didn't find good solution ===
    if best_primes is None or len(best_primes) > 5:
        # Build pool of 5-smooth numbers
        pool = set()
        for a in range(0, 15):
            for b in range(0, 15):
                for c in range(0, 15):
                    val = (2 ** a) * (3 ** b) * (5 ** c)
                    if 1 <= val <= MAX_VAL:
                        pool.add(val)
        pool = sorted(pool)
        
        if len(pool) >= N:
            for restart in range(30):
                if time.time() > deadline - 5.0:
                    break
                cur_x = random.sample(pool[:200], N)
                cur_primes = evaluate(cur_x)
                update_best(cur_x, cur_primes)
                
                improved = True
                iterations = 0
                while improved and iterations < 100 and time.time() < deadline - 5.0:
                    improved = False
                    iterations += 1
                    indices = list(range(N))
                    random.shuffle(indices)
                    
                    for i in indices:
                        if time.time() > deadline - 5.0:
                            break
                        old_x = cur_x[i]
                        idx = pool.index(old_x)
                        for next_idx in range(max(0, idx - 10), min(len(pool), idx + 50)):
                            new_x = pool[next_idx]
                            if new_x == old_x or new_x in cur_x:
                                continue
                            cur_x[i] = new_x
                            new_primes = evaluate(cur_x)
                            if len(new_primes) < len(cur_primes):
                                cur_primes = new_primes
                                improved = True
                                break
                            cur_x[i] = old_x
                        if improved:
                            update_best(cur_x, cur_primes)
                            break
    
    # === General random search as last resort ===
    if best_primes is None or len(best_primes) > 6:
        for restart in range(100):
            if time.time() > deadline - 2.0:
                break
            
            a = random.sample(range(1, MAX_VAL + 1), N)
            current_primes = evaluate(a)
            update_best(a, current_primes)
            
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                indices = list(range(N))
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() > deadline - 2.0:
                        break
                    
                    old_val = a[i]
                    for _ in range(10):
                        new_val = random.randint(1, MAX_VAL)
                        if new_val in a:
                            continue
                        
                        a[i] = new_val
                        new_primes = evaluate(a)
                        
                        if len(new_primes) < len(current_primes):
                            current_primes = new_primes
                            improved = True
                            update_best(a, current_primes)
                            break
                        a[i] = old_val
                    
                    if improved:
                        break
    
    if best_set:
        return best_set
    
    # Fallback
    return list(range(1, N+1))
