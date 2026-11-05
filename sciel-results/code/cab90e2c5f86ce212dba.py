import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill
    
    # Precomputation
    N = 511
    
    # Sieve for primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime

    is_prime = sieve(511)
    primes = [p for p in range(2, 512) if is_prime[p]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Prime factor masks for each number 1..511
    # mask[a] is a Python int (bitmask) of prime indices
    masks = [0] * 512
    for a in range(2, 512):
        m = a
        mask_val = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask_val |= (1 << prime_to_idx[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            # m is a prime
            mask_val |= (1 << prime_to_idx[m])
        masks[a] = mask_val
    masks[1] = 0 # 1 is coprime to everything, no prime factors

    # Weights
    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512 - a)
        
    # Candidates sorted by weight descending
    # We only care about 1..511
    candidates = list(range(511, 0, -1))
    candidates.sort(key=lambda x: weights[x], reverse=True)
    
    # 1. Greedy Initial Solution
    def greedy_solve():
        used_primes = 0
        score = 0.0
        sol = []
        for a in candidates:
            if (masks[a] & used_primes) == 0:
                sol.append(a)
                used_primes |= masks[a]
                score += weights[a]
        return sol, score

    initial_sol, initial_score = greedy_solve()
    best_score = initial_score
    best_sol = initial_sol[:]
    
    # 2. Local Search with 2-for-1 swaps (from inspiration)
    def local_search(sol):
        sol = sol[:]
        score = sum(weights[a] for a in sol)
        improved = True
        # Limit iterations to avoid getting stuck
        max_iters = 200
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            if time.time() - start_time > DEADLINE - 10:
                break

            # Try to add any compatible number
            used = 0
            for a in sol:
                used |= masks[a]
            
            added = False
            for a in candidates:
                if a in sol:
                    continue
                if (masks[a] & used) == 0:
                    sol.append(a)
                    score += weights[a]
                    used |= masks[a]
                    improved = True
                    added = True
                    break # Restart loop
            
            if added:
                continue

            # Try 1-for-1 swap
            swap_made = False
            for i in range(len(sol)):
                a_out = sol[i]
                remaining = [x for j, x in enumerate(sol) if j != i]
                rem_used = 0
                for a in remaining:
                    rem_used |= masks[a]
                
                best_swap = -1
                best_gain = 1e-9 # Small epsilon
                for a_in in candidates:
                    if a_in in remaining:
                        continue
                    if (masks[a_in] & rem_used) == 0:
                        gain = weights[a_in] - weights[a_out]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = a_in
                
                if best_swap != -1:
                    sol[i] = best_swap
                    score += best_gain
                    improved = True
                    swap_made = True
                    break # Restart loop
            
            if swap_made:
                continue

            # Try 2-for-1 swap: replace two numbers a1, a2 with one number a3
            # This is useful when a3 is large (high weight) but conflicts with both a1 and a2,
            # while a1 and a2 are small (low weight).
            # Actually, the insight says "replace primes 17 and 29 with 493".
            # 493 = 17 * 29.
            # Weight(493) = 1/(512-493) = 1/19 ~ 0.0526
            # Weight(17) = 1/(512-17) = 1/495 ~ 0.002
            # Weight(29) = 1/(512-29) = 1/483 ~ 0.002
            # Sum weights of 17, 29 is ~ 0.004.
            # So replacing two small numbers with one large composite is a huge gain.
            # The condition is that a3 is coprime to all OTHER numbers in the solution.
            # And a3 conflicts with a1 and a2 (which is always true if a3 is a multiple of their primes, 
            # unless a1/a2 are 1, but 1 has no primes).
            
            found_2for1 = False
            # Iterate over pairs in sol
            for i in range(len(sol)):
                for j in range(i + 1, len(sol)):
                    a1 = sol[i]
                    a2 = sol[j]
                    
                    # Calculate used primes by rest of solution
                    rem_used = 0
                    for k in range(len(sol)):
                        if k != i and k != j:
                            rem_used |= masks[sol[k]]
                    
                    rem_score = score - weights[a1] - weights[a2]
                    
                    best_swap_in = -1
                    best_gain = 1e-9
                    for a_in in candidates:
                        if a_in == a1 or a_in == a2:
                            continue
                        if a_in in sol:
                            continue
                        # Must be coprime to rest
                        if (masks[a_in] & rem_used) != 0:
                            continue
                        
                        # It's okay if it conflicts with a1 or a2, we are removing them.
                        # But we must ensure it's distinct.
                        
                        gain = weights[a_in] - weights[a1] - weights[a2]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap_in = a_in
                    
                    if best_swap_in != -1:
                        # Remove a1, a2, add best_swap_in
                        # To keep list order somewhat stable, just modify in place
                        # sol[i] = best_swap_in
                        # del sol[j]
                        # This changes indices. Let's rebuild.
                        new_sol = []
                        for k in range(len(sol)):
                            if k != i and k != j:
                                new_sol.append(sol[k])
                        new_sol.append(best_swap_in)
                        sol = new_sol
                        score += best_gain
                        improved = True
                        found_2for1 = True
                        break # Restart loop
                if found_2for1:
                    break
            
            # Try 1-for-2 swap: replace one number a1 with two numbers a2, a3
            # This is less likely to be beneficial given the convexity, but possible
            # if a1 is small and a2, a3 are large but coprime to each other and rest.
            # However, 1/(512-a) is convex, so sum of two smaller terms is usually less than one large term?
            # 1/x is convex. So f(x1)+f(x2) >= 2f((x1+x2)/2).
            # But here we are comparing 1/(512-a1) vs 1/(512-a2) + 1/(512-a3).
            # If a1 is small (e.g., 1), weight is small.
            # If a2, a3 are large (e.g., 500, 501), weights are large.
            # So 1-for-2 could be beneficial.
            
            # Let's add 1-for-2 swap
            found_1for2 = False
            for i in range(len(sol)):
                a1 = sol[i]
                rem_used = 0
                for k in range(len(sol)):
                    if k != i:
                        rem_used |= masks[sol[k]]
                
                rem_score = score - weights[a1]
                
                # Find best pair a2, a3 such that:
                # 1. Coprime to rem_used
                # 2. Coprime to each other
                # 3. Distinct from a1
                # 4. Maximize weights[a2] + weights[a3] - weights[a1]
                
                # Optimization: Pre-filter candidates compatible with rem_used
                compat = []
                for a in candidates:
                    if a == a1:
                        continue
                    if (masks[a] & rem_used) == 0:
                        compat.append(a)
                
                # Find best pair in compat
                best_pair = None
                best_sum = -1e9
                # compat is sorted by weight desc
                # We can limit search to top few?
                # For now, check all pairs in compat.
                # To speed up, if len(compat) is large, this is O(N^2).
                # N is 511. 511^2 ~ 260k. Acceptable per iteration if not too many iterations.
                # But we are in a loop.
                # Let's just check top 50 in compat to save time.
                top_compat = compat[:50]
                for x in range(len(top_compat)):
                    a2 = top_compat[x]
                    for y in range(x + 1, len(top_compat)):
                        a3 = top_compat[y]
                        if (masks[a2] & masks[a3]) == 0:
                            s = weights[a2] + weights[a3]
                            if s > best_sum:
                                best_sum = s
                                best_pair = (a2, a3)
                
                if best_pair:
                    gain = best_sum - weights[a1]
                    if gain > 1e-9:
                        new_sol = []
                        for k in range(len(sol)):
                            if k != i:
                                new_sol.append(sol[k])
                        new_sol.append(best_pair[0])
                        new_sol.append(best_pair[1])
                        sol = new_sol
                        score += gain
                        improved = True
                        found_1for2 = True
                        break # Restart loop
            
            if found_1for2:
                continue

        return sol, score

    sol, score = local_search(initial_sol)
    if score > best_score:
        best_score = score
        best_sol = sol[:]

    # 3. Branch and Bound with Greedy Coloring Upper Bound
    # We will perform a final BNB to ensure optimality or find better local optima.
    # Since N=511, a full BNB is too slow. We will use a limited depth or time-limited BNB.
    # The "greedy coloring" bound on the conflict graph (non-coprime) corresponds to
    # finding a clique cover or independent set bound.
    # Actually, for Maximum Weight Independent Set, a common bound is based on
    # coloring the conflict graph. The number of colors is an upper bound on the size of the independent set.
    # For weighted, we can use a weighted coloring bound.
    # However, implementing a tight weighted coloring bound is complex.
    # Given the time limit, the local search with 2-for-1 swaps is likely the most impactful improvement.
    # The current BNB in the seed program was weak.
    # Let's stick with the improved local search.
    
    # One more pass of local search with different initial order or perturbation?
    # Let's try a simple random restart local search if time permits.
    if time.time() - start_time < DEADLINE - 20:
        import random
        random.seed(42)
        
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Perturb the best solution
            perturbed_sol = best_sol[:]
            # Remove 1-3 random elements
            for _ in range(random.randint(1, 3)):
                if perturbed_sol:
                    idx = random.randint(0, len(perturbed_sol) - 1)
                    perturbed_sol.pop(idx)
            
            # Run local search on perturbed solution
            p_sol, p_score = local_search(perturbed_sol)
            if p_score > best_score:
                best_score = p_score
                best_sol = p_sol[:]
                
    return best_sol
