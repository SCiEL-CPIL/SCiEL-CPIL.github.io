import math
import random
import time

def construct():
    deadline = time.time() + 115.0
    N = 511
    C = 512

    # Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Weights
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)
        
    # Precompute conflicts as sets
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if math.gcd(a, b) > 1:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    # Pre-sort indices by weight (descending)
    # a=511 has highest weight, a=1 has lowest
    sorted_indices = list(range(N, 0, -1))
    
    def can_add(a, S_set):
        return not (S_set & conflicts[a])
        
    def get_score(S_set):
        return sum(w[a] for a in S_set)
        
    def find_best_add(S_set):
        # Iterate through pre-sorted indices
        for a in sorted_indices:
            if a in S_set:
                continue
            if can_add(a, S_set):
                return a
        return None

    def greedy_init(rng=None):
        S = set()
        # Always add 1? 1 is coprime to everything.
        # w[1] = 1/511 ~ 0.0019
        # It might be better to skip 1 if it blocks nothing, but it doesn't block anything.
        # So adding 1 is always good? 
        # Yes, 1 is coprime to all. So 1 should always be in the set.
        S.add(1)
        
        candidates = sorted_indices.copy()
        if rng:
            # Perturb the order slightly or shuffle a subset
            # Actually, let's try different strategies
            pass
            
        # Strategy: Greedy by weight
        for a in candidates:
            if a in S:
                continue
            if can_add(a, S):
                S.add(a)
                
        return S

    def local_search(S, max_iter=100, rng=None):
        if rng is None:
            rng = random.Random(42)
            
        current_score = get_score(S)
        best_S = S.copy()
        best_score = current_score
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time.time() < deadline - 5.0:
            improved = False
            iteration += 1
            S_list = list(S)
            rng.shuffle(S_list)
            
            # Try 1-swap (remove x, add y)
            # Only try removing x if there is a better y
            for x in S_list:
                if time.time() > deadline - 5.0:
                    break
                new_S = S - {x}
                y = find_best_add(new_S)
                if y is not None and w[y] > w[x]:
                    delta = w[y] - w[x]
                    S = new_S | {y}
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S.copy()
                    improved = True
                    break
                        
            if improved:
                continue

            # Try 2-swap (remove x, y, add u, v)
            # Optimize: Only try pairs (x, y) where x, y are "small" (low weight)
            # Sort S by weight ascending to try removing the worst elements first
            S_sorted = sorted(S, key=lambda a: w[a])
            
            found_move = False
            for i in range(len(S_sorted)):
                if time.time() > deadline - 5.0:
                    break
                x = S_sorted[i]
                for j in range(i + 1, len(S_sorted)):
                    y = S_sorted[j]
                    new_S = S - {x, y}
                    u = find_best_add(new_S)
                    if u is None:
                        continue
                    new_S_u = new_S | {u}
                    v = find_best_add(new_S_u)
                    if v is None:
                        continue
                    delta = w[u] + w[v] - w[x] - w[y]
                    if delta > 1e-9:
                        S = new_S | {u, v}
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = S.copy()
                        improved = True
                        found_move = True
                        break
                if found_move:
                    break
                    
            if not found_move:
                # Try random 2-swap to escape local minima?
                # Or try 1-remove-2-add
                found_move = False
                for x in S_list:
                    if time.time() > deadline - 5.0:
                        break
                    new_S = S - {x}
                    # Find best u
                    u = find_best_add(new_S)
                    if u is None:
                        continue
                    new_S_u = new_S | {u}
                    v = find_best_add(new_S_u)
                    if v is None:
                        continue
                    delta = w[u] + w[v] - w[x]
                    if delta > 1e-9:
                        S = new_S_u | {v}
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = S.copy()
                        improved = True
                        found_move = True
                        break
                        
        return best_S, best_score

    best_overall_S = set()
    best_overall_score = 0.0
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy
    S1 = greedy_init(rng)
    S1, score1 = local_search(S1, max_iter=200, rng=rng)
    if score1 > best_overall_score:
        best_overall_score = score1
        best_overall_S = S1.copy()
        
    # Strategy 2: Random Restarts
    # We can afford more restarts if local search is fast
    for restart in range(15):
        if time.time() > deadline - 10.0:
            break
        S_r = greedy_init(rng)
        # Add some randomness to the greedy order?
        # For now, just run local search
        S_r, score_r = local_search(S_r, max_iter=100, rng=rng)
        if score_r > best_overall_score:
            best_overall_score = score_r
            best_overall_S = S_r.copy()
            
    # Final polish
    for a in sorted_indices:
        if a not in best_overall_S and can_add(a, best_overall_S):
            best_overall_S.add(a)
            best_overall_score += w[a]
            
    # One more local search
    best_overall_S, best_overall_score = local_search(best_overall_S, max_iter=50, rng=rng)
    
    return list(best_overall_S)
