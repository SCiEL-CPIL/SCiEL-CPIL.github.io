"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute smallest prime factor for fast factorization
    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def factorize(n):
        """Return set of prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def evaluate(arr):
        """Return the number of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)

    def get_prime_set(arr):
        """Return the set of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return primes

    def is_P_smooth(n, P):
        """Check if all prime factors of n are in set P."""
        if n <= 1:
            return True
        temp = n
        while temp > 1:
            p = spf[temp]
            if p not in P:
                return False
            while temp % p == 0:
                temp //= p
        return True

    # Strategy:
    # 1. Deterministic initialization using modular arithmetic (multiples of 210) to ensure
    #    all sums are divisible by 2, 3, 5, 7. This provides a strong starting point.
    # 2. Constraint Satisfaction / Backtracking Search for specific small prime sets P.
    #    We try to find a set of 12 numbers such that all pairwise sums are P-smooth.
    #    We use forward checking to prune candidates based on incompatibilities.
    # 3. Local Search refinement if exact P-smooth sets are not found or to improve further.

    # --- Phase 1: Deterministic Initialization ---
    # Multiples of 210 ensure sums are divisible by 2, 3, 5, 7.
    # Let's pick 12 multiples of 210.
    # 210, 420, ..., 210*12 = 2520. All <= 10000.
    init_set = [210 * i for i in range(1, 13)]
    best = init_set
    best_score = evaluate(best)
    
    # Also try a denser set of smaller multiples of 6 or 12 to see if lower primes appear
    # But 210 is a good start because it covers small primes.
    # Let's also try multiples of 6.
    init_set_6 = [6 * i for i in range(1, 13)]
    score_6 = evaluate(init_set_6)
    if score_6 < best_score:
        best = init_set_6
        best_score = score_6

    # --- Phase 2: Backtracking Search for P-Smooth Sets ---
    # We want to find 12 numbers such that all pairwise sums have prime factors ONLY from P.
    # If we find such a set, the score is |P|.
    # We try small P sets.
    
    # Candidate P sets to try (sorted by size, then by value)
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 5},
        {2, 7},
        {3, 5},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 5, 7},
    ]

    # Limit search space to small numbers for the backtracking to be feasible.
    # If numbers are small, sums are small, more likely to be smooth.
    # Let's search in range 1..200 for P-smooth sets.
    SEARCH_RANGE = 200
    
    for P in prime_sets_to_try:
        if len(P) >= best_score:
            continue # Can't improve
        
        # Generate candidates: numbers 1..SEARCH_RANGE
        # We can filter candidates: x is a candidate only if x+y is P-smooth for some y?
        # Actually, we need x+y to be P-smooth for ALL other chosen y's.
        # Let's just use all numbers 1..SEARCH_RANGE as potential candidates.
        candidates = list(range(1, SEARCH_RANGE + 1))
        
        # Precompute compatibility:
        # compat[i][j] is True if candidates[i] + candidates[j] is P-smooth
        # Note: i != j
        # To save memory/time, we can compute on the fly or use a set of valid sums.
        # Let's precompute a set of "valid sums" for this P.
        # A sum S is valid if is_P_smooth(S, P).
        # Max sum = 400.
        valid_sums = set()
        for s in range(2, 2 * SEARCH_RANGE + 1):
            if is_P_smooth(s, P):
                valid_sums.add(s)
        
        # Backtracking
        def backtrack(current_set, remaining_candidates, depth):
            if time.time() > deadline:
                return None
            
            if depth == 12:
                return list(current_set)
            
            # Forward checking: 
            # For each candidate in remaining_candidates, check if it's compatible with all in current_set
            # And if there are enough remaining candidates compatible with it to fill the rest.
            
            # Optimization: Sort candidates by degree (number of compatible partners) to fail fast?
            # Or just iterate.
            
            # We need to pick one candidate from remaining_candidates.
            # Let's iterate through them.
            # To improve pruning, we can check if the intersection of compatible sets for all current elements
            # has size >= remaining slots.
            
            if not remaining_candidates:
                return None
            
            # Calculate the set of candidates that are compatible with ALL elements in current_set
            # If current_set is empty, all candidates are compatible.
            if not current_set:
                compatible = set(remaining_candidates)
            else:
                # Start with first element's compatible set
                first_elem = next(iter(current_set))
                compatible = {c for c in remaining_candidates if (first_elem + c) in valid_sums}
                for elem in current_set:
                    if elem == first_elem:
                        continue
                    # Intersect with elem's compatible set
                    # This is expensive. Let's precompute adjacency lists for speed.
                    pass
            
            # Let's restructure: Precompute adjacency for the current P.
            # adj[x] = set of y in 1..SEARCH_RANGE such that x+y is P-smooth.
            # We can build this once per P.
            
            return None # Placeholder, will implement properly below

        # Build adjacency list for this P
        # adj[x] is a set of y such that x+y is in valid_sums
        adj = [set() for _ in range(SEARCH_RANGE + 1)]
        for x in range(1, SEARCH_RANGE + 1):
            for y in range(1, SEARCH_RANGE + 1):
                if x != y and (x + y) in valid_sums:
                    adj[x].add(y)
        
        # Backtracking with forward checking
        def bt_search(chosen, candidates_mask):
            """
            chosen: list of chosen numbers
            candidates_mask: bitmask or set of available candidates
            """
            if time.time() > deadline:
                return None
            
            if len(chosen) == 12:
                return list(chosen)
            
            # Forward checking:
            # The next candidate must be compatible with all chosen.
            # Let C be the set of candidates compatible with all chosen.
            if not chosen:
                C = set(candidates_mask)
            else:
                # Intersection of adj[x] for all x in chosen, intersected with candidates_mask
                # Start with first
                C = adj[chosen[0]] & candidates_mask
                for x in chosen[1:]:
                    C = C & adj[x]
                    if not C:
                        return None
            
            if not C:
                return None
            
            # Heuristic: Pick candidate with highest degree in C?
            # Or just iterate.
            # To avoid symmetry, we can enforce an ordering? 
            # Not strictly necessary if we just want ANY solution.
            # But to avoid re-visiting equivalent states, we can sort.
            
            # Let's try candidates in C.
            # We can order them by how many future options they leave.
            # For simplicity, let's just iterate.
            
            # Optimization: If we pick x, the next candidates must be in C \ {x} AND compatible with x.
            # Which is C & adj[x] \ {x}.
            
            for x in C:
                new_candidates = C & adj[x]
                new_candidates.discard(x)
                
                # Check if new_candidates has enough elements to fill remaining slots?
                # len(chosen) + 1 + len(new_candidates) >= 12 ?
                if len(new_candidates) < 12 - (len(chosen) + 1):
                    continue
                
                res = bt_search(chosen + [x], new_candidates)
                if res:
                    return res
            
            return None

        # Start search
        # Initial candidates: all 1..SEARCH_RANGE
        # We can prune initial candidates: x must have degree >= 11
        initial_candidates = {x for x in range(1, SEARCH_RANGE + 1) if len(adj[x]) >= 11}
        
        # To make search deterministic and fast, sort initial candidates
        # We can try starting with the one with max degree
        if initial_candidates:
            # Pick a starting candidate
            start_cand = max(initial_candidates, key=lambda x: len(adj[x]))
            first_choice = [start_cand]
            new_cands = adj[start_cand] & initial_candidates
            new_cands.discard(start_cand)
            
            if len(new_cands) >= 11:
                res = bt_search(first_choice, new_cands)
                if res:
                    # Found a set of 12 numbers with prime factors subset of P
                    # Score is |P|
                    if len(P) < best_score:
                        best = res
                        best_score = len(P)
                        # We can stop if we hit the theoretical minimum? 
                        # Minimum is 1 (P={2}).
                        if best_score == 1:
                            return best

    # --- Phase 3: Local Search Refinement ---
    # If we haven't found a perfect P-smooth set, refine the best found so far.
    # Use the local search from the original program but with a better initial state.
    
    random.seed(42)
    
    # Generate candidate pool
    candidate_pool = []
    for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 210]:
        for k in range(1, 10000 // m + 1):
            val = k * m
            if 1 <= val <= 10000:
                candidate_pool.append(val)
    for _ in range(2000):
        candidate_pool.append(random.randint(1, 10000))
    candidate_pool = list(set(candidate_pool))
    random.shuffle(candidate_pool)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            old_val = best[i]
            for new_val in candidate_pool:
                if time.time() > deadline:
                    break
                if new_val == old_val:
                    continue
                if new_val in best:
                    continue
                candidate = best.copy()
                candidate[i] = new_val
                score = evaluate(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    break
            if time.time() > deadline:
                break

    return best

# EVOLVE-BLOCK-END
