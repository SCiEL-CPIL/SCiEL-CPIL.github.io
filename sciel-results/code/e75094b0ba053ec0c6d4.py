"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit
    
    # Helper to get prime factors using SPF for speed
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def prime_factors(n):
        factors = set()
        if n <= 1:
            return factors
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    # Precompute prime factors for all possible sums (2 to 20000)
    pf_cache = {}
    for s in range(2, MAX_SUM + 1):
        pf_cache[s] = frozenset(prime_factors(s))

    def evaluate_set(nums):
        all_primes = set()
        n = len(nums)
        for i in range(n):
            ai = nums[i]
            for j in range(i + 1, n):
                all_primes.update(pf_cache[ai + nums[j]])
        return len(all_primes)

    # Helper to check if n is P-smooth
    def is_p_smooth(n, primes):
        if n <= 0:
            return False
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    # Strategy 1: Structured Backtracking / Constraint Satisfaction
    # We want to find 12 distinct integers <= 10000 such that all pairwise sums
    # have prime factors from a small set P.
    # If we restrict P to {2, 3, 5, 7}, then all sums must be 7-smooth.
    # Let's try to find a set of 12 numbers where all pairwise sums are 7-smooth.
    
    P = [2, 3, 5, 7]
    
    # Generate 7-smooth numbers up to 20000 (max sum)
    # x + y is 7-smooth.
    # If we pick x_i from a specific structure, we might simplify.
    # Consider x_i = A * k_i + B.
    # Sum = A(k_i + k_j) + 2B.
    
    # Alternative: Just search for a clique in the graph where edge (i,j) exists if i+j is 7-smooth.
    # But the vertices are the numbers themselves (1..10000).
    # We need a clique of size 12 in this graph.
    
    # Let's generate a pool of candidate numbers.
    # Numbers that are "likely" to form smooth sums with many other numbers.
    # Multiples of 210 (2*3*5*7) are good candidates because 210*k is divisible by 2,3,5,7.
    # Sum of two multiples of 210: 210(k1+k2). This is 7-smooth iff k1+k2 is 7-smooth.
    # So the problem reduces to finding 12 distinct k_i <= 10000/210 ≈ 47 such that
    # all pairwise sums k_i + k_j are 7-smooth.
    
    max_k = 10000 // 210
    # Generate 7-smooth numbers for k sums
    # k_i + k_j <= 2 * 47 = 94.
    # 7-smooth numbers up to 100:
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 75, 80, 81, 84, 90, 96, 98, 100...
    
    smooth_sums = set()
    def gen_smooth(max_val, primes):
        if not primes:
            if max_val >= 1:
                yield 1
            return
        p = primes[0]
        rest = primes[1:]
        pow_p = 1
        while pow_p <= max_val:
            for val in gen_smooth(max_val // pow_p, rest):
                yield val * pow_p
            pow_p *= p

    for s in gen_smooth(100, P):
        smooth_sums.add(s)
    
    # Now, find 12 distinct k in [1, max_k] such that for all i != j, k_i + k_j in smooth_sums.
    # This is a clique finding problem on a small graph (size ~47).
    
    k_candidates = list(range(1, max_k + 1))
    n_k = len(k_candidates)
    
    # Build adjacency matrix for k's
    adj_k = [[False] * n_k for _ in range(n_k)]
    for i in range(n_k):
        for j in range(i + 1, n_k):
            if k_candidates[i] + k_candidates[j] in smooth_sums:
                adj_k[i][j] = True
                adj_k[j][i] = True
                
    # Bron-Kerbosch with pivot to find clique of size 12
    clique = []
    found_clique = False
    
    def bron_kerbosch(R, P_set, X):
        nonlocal found_clique, clique
        if time.time() > deadline:
            return
        if len(R) == 12:
            found_clique = True
            clique = R.copy()
            return
        if not P_set and not X:
            return
            
        # Pivot selection
        if P_set or X:
            U = list(P_set) + list(X)
            u = max(U, key=lambda v: sum(adj_k[v][w] for w in (P_set | X)))
            candidates = [v for v in P_set if not adj_k[v][u]]
        else:
            candidates = list(P_set)
            
        for v in candidates:
            bron_kerbosch(R + [v], {w for w in P_set if adj_k[v][w]}, {w for w in X if adj_k[v][w]})
            if found_clique:
                return
            P_set.remove(v)
            X.add(v)
            
    # Initial P_set: indices of k's that have degree >= 11
    initial_P = {i for i in range(n_k) if sum(adj_k[i]) >= 11}
    if initial_P:
        bron_kerbosch([], initial_P, set())
        
    if found_clique:
        result = [210 * k_candidates[k] for k in clique]
        if len(set(result)) == 12 and all(x <= 10000 for x in result):
            # Verify
            if evaluate_set(result) <= 4: # Should be exactly primes in P
                return result

    # Strategy 2: Try other bases.
    # What if we use base 30 (2*3*5) and allow prime 7 in sums?
    # x = 30 * k. Sum = 30(k1+k2).
    # Primes are {2,3,5} U Primes(k1+k2).
    # We want Primes(k1+k2) subset {7}. So k1+k2 must be 7-smooth.
    # k_i <= 10000/30 = 333.
    # Sums <= 666.
    
    max_k2 = 10000 // 30
    smooth_sums2 = set()
    for s in gen_smooth(700, P): # 7-smooth up to 700
        smooth_sums2.add(s)
        
    k_candidates2 = list(range(1, max_k2 + 1))
    n_k2 = len(k_candidates2)
    
    adj_k2 = [[False] * n_k2 for _ in range(n_k2)]
    for i in range(n_k2):
        for j in range(i + 1, n_k2):
            if k_candidates2[i] + k_candidates2[j] in smooth_sums2:
                adj_k2[i][j] = True
                adj_k2[j][i] = True
                
    found_clique2 = False
    clique2 = []
    
    def bron_kerbosch2(R, P_set, X):
        nonlocal found_clique2, clique2
        if time.time() > deadline:
            return
        if len(R) == 12:
            found_clique2 = True
            clique2 = R.copy()
            return
        if not P_set and not X:
            return
            
        if P_set or X:
            U = list(P_set) + list(X)
            u = max(U, key=lambda v: sum(adj_k2[v][w] for w in (P_set | X)))
            candidates = [v for v in P_set if not adj_k2[v][u]]
        else:
            candidates = list(P_set)
            
        for v in candidates:
            bron_kerbosch2(R + [v], {w for w in P_set if adj_k2[v][w]}, {w for w in X if adj_k2[v][w]})
            if found_clique2:
                return
            P_set.remove(v)
            X.add(v)
            
    initial_P2 = {i for i in range(n_k2) if sum(adj_k2[i]) >= 11}
    if initial_P2:
        bron_kerbosch2([], initial_P2, set())
        
    if found_clique2:
        result = [30 * k_candidates2[k] for k in clique2]
        if len(set(result)) == 12 and all(x <= 10000 for x in result):
            if evaluate_set(result) <= 4:
                return result

    # Strategy 3: Local Search with Structured Initialization
    # If exact clique search fails or takes too long, fall back to local search.
    
    best_score = float('inf')
    best_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    best_score = evaluate_set(best_set)
    
    # Seeds
    seeds = [
        [210 * i for i in range(1, 13)],
        [30 * i for i in range(1, 13)],
        [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30],
        [2 * i for i in range(1, 13)],
        [2 * i + 1 for i in range(1, 13)],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    ]
    
    for s in seeds:
        if len(set(s)) == 12 and all(1 <= x <= 10000 for x in s):
            score = evaluate_set(s)
            if score < best_score:
                best_score = score
                best_set = s.copy()
                if best_score <= 4:
                    return best_set

    # Local Search
    def local_search(current, curr_score, max_iters=2000):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            idx = random.randint(0, 11)
            old_val = current[idx]
            
            # Generate candidates
            candidates = []
            # Small perturbations
            for delta in [-2, -1, 1, 2, 5, 10, 30, 210]:
                candidates.append(old_val + delta)
                candidates.append(old_val - delta)
            
            # Random restarts
            candidates.append(random.randint(1, 10000))
            
            # Bias towards smooth numbers
            if random.random() < 0.5:
                candidates.append(210 * random.randint(1, 47))
            if random.random() < 0.5:
                candidates.append(30 * random.randint(1, 333))
                
            for new_val in candidates:
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in current and new_val != old_val:
                    continue
                
                new_set = current.copy()
                new_set[idx] = new_val
                
                new_score = evaluate_set(new_set)
                
                if new_score < curr_score:
                    current = new_set
                    curr_score = new_score
                    improved = True
                    break
                    
        return current, curr_score

    # Run local search from best seed
    for _ in range(10):
        if time.time() > deadline:
            break
            
        if _ == 0:
            curr = best_set.copy()
            curr_score = best_score
        else:
            # Perturb best
            curr = best_set.copy()
            for _ in range(2):
                idx = random.randint(0, 11)
                curr[idx] = random.randint(1, 10000)
            # Ensure distinct
            while len(set(curr)) < 12:
                curr[random.randint(0, 11)] = random.randint(1, 10000)
            curr_score = evaluate_set(curr)
            
        curr, curr_score = local_search(curr, curr_score)
        
        if curr_score < best_score:
            best_score = curr_score
            best_set = curr.copy()
            if best_score <= 4:
                return best_set

    return best_set
# EVOLVE-BLOCK-END
