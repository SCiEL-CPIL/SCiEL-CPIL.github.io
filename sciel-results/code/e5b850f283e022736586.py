import math
import time
import random

def construct():
    deadline = time.time() + 145.0
    N = 511
    C = 512

    # Weights: w[a] = 1 / (C - a)
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)

    # Precompute prime factors for all numbers 1..511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(C)
    prime_factors = [set() for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            prime_factors[multiple].add(p)
    
    # 1 is coprime to everything, so prime_factors[1] = empty set
    prime_factors[1] = set()

    def is_coprime_set(a, S):
        """Check if a is coprime to all elements in S."""
        fa = prime_factors[a]
        if not fa:
            return True
        for b in S:
            if fa & prime_factors[b]:
                return False
        return True

    # Greedy Initial Solution
    # Sort by weight descending (largest a first)
    candidates = list(range(N, 0, -1))
    S = set()
    current_score = 0.0
    for a in candidates:
        if is_coprime_set(a, S):
            S.add(a)
            current_score += w[a]

    best_S = set(S)
    best_score = current_score

    def local_search(S, current_score):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1-Add: Try to add any node
            # We can only add if it's coprime to all in S
            # This is rare if S is large, but possible if S has gaps
            # Actually, if we started with greedy, we likely can't add.
            # But after swaps, maybe.
            for a in range(N, 0, -1):
                if a in S:
                    continue
                if is_coprime_set(a, S):
                    S.add(a)
                    current_score += w[a]
                    improved = True
                    break
            
            if not improved:
                # 1-Swap: Try to replace one element
                # Iterate over elements in S, sorted by weight ascending (remove low value first)
                # But we need to check all potential replacements.
                # To be efficient, we can iterate over potential replacements first?
                # No, for each x in S, we need to find y not in S such that w[y] > w[x] and y coprime to S\{x}.
                
                # Optimization: Precompute "conflict" with S\{x}
                # For each x in S, let R = S \ {x}.
                # We want y not in S, w[y] > w[x], and y coprime to R.
                
                # Let's iterate x in S
                for x in list(S):
                    if time.time() > deadline - 5.0:
                        break
                    R = S - {x}
                    # Find best y
                    best_y = -1
                    best_w_y = w[x]
                    
                    # Iterate y from N down to 1
                    # We can skip y if w[y] <= w[x]
                    for y in range(N, 0, -1):
                        if y in S:
                            continue
                        if w[y] <= best_w_y:
                            # Since we are going down, if w is not monotonic with index?
                            # w[y] = 1/(512-y). As y increases, w[y] increases.
                            # So w is monotonic increasing with y.
                            # If w[y] <= w[x], then for all smaller y, w is even smaller.
                            # So we can break?
                            # Yes, w is strictly increasing with a.
                            break
                        
                        if is_coprime_set(y, R):
                            best_y = y
                            best_w_y = w[y]
                            # Since we want max w, and we are going down from N, 
                            # the first valid y is the best y.
                            break
                    
                    if best_y != -1:
                        S.remove(x)
                        S.add(best_y)
                        current_score += w[best_y] - w[x]
                        improved = True
                        break # Restart loop as S changed
                
                if not improved:
                    # 2-Swap: Try to replace two elements
                    # This is O(|S|^2 * N) which is heavy.
                    # |S| is roughly 100-200?
                    # N=511.
                    # 200^2 * 511 ~ 20 million ops. Might be slow in Python.
                    # Let's try a limited 2-swap:
                    # Pick two lowest weight elements x1, x2.
                    # Try to replace them with two highest weight elements y1, y2 not in S.
                    
                    # Sort S by weight ascending
                    S_list = list(S)
                    S_list.sort(key=lambda x: w[x])
                    
                    # Try swapping the two lowest
                    if len(S_list) >= 2:
                        x1, x2 = S_list[0], S_list[1]
                        R = S - {x1, x2}
                        
                        # Find best pair (y1, y2) not in S, y1 != y2
                        # such that w[y1]+w[y2] > w[x1]+w[x2]
                        # and y1, y2 coprime to R and each other.
                        
                        # We can iterate y1 from N down to 1
                        # For each y1, iterate y2 from y1-1 down to 1
                        # This is O(N^2) per 2-swap attempt.
                        # 511^2 ~ 260k.
                        # If we do this a few times, it's fine.
                        
                        target_score = w[x1] + w[x2]
                        best_pair = None
                        best_pair_score = target_score
                        
                        # Optimize: Only consider y with w[y] > min(w[x1], w[x2])?
                        # Actually, we need sum > target.
                        
                        # Let's just scan from top.
                        # Since w is increasing, high y are good.
                        # We can restrict y1, y2 to be > some threshold?
                        # Let's just scan.
                        
                        for y1 in range(N, 0, -1):
                            if y1 in S:
                                continue
                            if w[y1] <= target_score: 
                                # Even with y2=0, we can't beat target? 
                                # No, w[y2] > 0. So if w[y1] is small, we might still beat it with a large y2?
                                # But we are scanning y1 from top.
                                # If w[y1] is already small, and we need sum > target,
                                # we need a very large y2.
                                # But y2 < y1 in our loop? No, we can just pick any pair.
                                # Let's iterate y1 from N to 1, and for each y1, iterate y2 from N to 1, y2 != y1.
                                # To avoid duplicates, y2 < y1.
                                # If w[y1] + w[N] <= target, then no y2 can help.
                                # So if w[y1] + w[N] <= target, break?
                                # Yes, because w[y2] <= w[N].
                                break
                            
                            if not is_coprime_set(y1, R):
                                continue
                            
                            for y2 in range(y1 - 1, 0, -1):
                                if y2 in S:
                                    continue
                                if w[y1] + w[y2] <= target_score:
                                    break
                                
                                if not is_coprime_set(y2, R):
                                    continue
                                
                                if prime_factors[y1] & prime_factors[y2]:
                                    continue
                                
                                # Valid pair
                                S.remove(x1)
                                S.remove(x2)
                                S.add(y1)
                                S.add(y2)
                                current_score += w[y1] + w[y2] - target_score
                                improved = True
                                best_pair = (y1, y2)
                                break
                            
                            if improved:
                                break
                        
                        if not improved:
                            # No 2-swap found with the lowest two.
                            # We could try other pairs, but it's expensive.
                            # Let's stop.
                            pass

        return S, current_score

    # Run local search
    S, current_score = local_search(S, current_score)
    
    if current_score > best_score:
        best_score = current_score
        best_S = set(S)

    # Random Restarts with Local Search
    # If time permits, try a few random starts
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        # Randomized Greedy Start
        random_S = set()
        random_score = 0.0
        # Shuffle candidates
        candidates = list(range(N, 0, -1))
        random.shuffle(candidates)
        # To bias towards high weights, we can sort by weight with noise
        # Or just shuffle and pick.
        # Better: Sort by weight, but with probability p, skip.
        # Let's just use a perturbed greedy.
        
        # Alternative: Start from best_S and do random swaps to escape local optima
        # This is "Simulated Annealing" like.
        
        # Let's try a simple random restart:
        # Pick a random subset of best_S to keep, and fill rest with greedy?
        
        # Simpler: Randomly perturb best_S
        perturbed_S = set(best_S)
        perturbed_score = best_score
        
        # Remove k random elements
        k = random.randint(2, 5)
        if len(perturbed_S) > k:
            to_remove = random.sample(list(perturbed_S), k)
            for x in to_remove:
                perturbed_S.remove(x)
                perturbed_score -= w[x]
        
        # Run local search on perturbed_S
        S_new, score_new = local_search(perturbed_S, perturbed_score)
        
        if score_new > best_score:
            best_score = score_new
            best_S = S_new
        else:
            # If no improvement, maybe try a different perturbation
            pass

    # Final Polish: Try to add any missing coprime numbers
    for a in range(N, 0, -1):
        if a not in best_S and is_coprime_set(a, best_S):
            best_S.add(a)
            best_score += w[a]

    return list(best_S)
