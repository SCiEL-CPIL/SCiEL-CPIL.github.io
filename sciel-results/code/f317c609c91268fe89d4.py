import time
from math import gcd

def construct():
    deadline = time.time() + 55.0
    start = time.time()
    
    # --- Precomputation ---
    # multiples_of[k] = list of m in [2,256] such that m % k == 0
    multiples_of = [[] for _ in range(257)]
    for k in range(1, 257):
        for m in range(k, 257, k):
            if m >= 2:
                multiples_of[k].append(m)
                
    # Precompute largest prime factor for ordering heuristics
    lpf = [0] * 257
    for i in range(2, 257):
        if lpf[i] == 0: # i is prime
            for j in range(i, 257, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    candidates = list(range(2, 257))
    primes_large = [p for p in candidates if lpf[p] == p and p > 128]
    candidates_rest = [x for x in candidates if x not in primes_large]

    def can_add(a, in_set):
        """
        Check if a can be added to the set represented by in_set (boolean array/list).
        Returns True if valid, False otherwise.
        """
        for b in range(2, 257):
            if not in_set[b]:
                continue
            
            g = gcd(a, b)
            
            # Check 1: a | b*c => (a/g) | c
            k1 = a // g
            # We need c in S, c != b, such that c is a multiple of k1
            for c in multiples_of[k1]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
            
            # Check 2: b | a*c => (b/g) | c
            k2 = b // g
            for c in multiples_of[k2]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
                    
        return True

    def can_add_antichain(a, in_set):
        """
        Check if a can be added to in_set, assuming in_set is an antichain and we want to maintain it.
        Also checks the product divisibility constraints.
        """
        # 1. Antichain check
        for b in range(2, 257):
            if not in_set[b]:
                continue
            if a % b == 0 or b % a == 0:
                return False
        
        # 2. Product divisibility check (same as before)
        for b in range(2, 257):
            if not in_set[b]:
                continue
            
            g = gcd(a, b)
            
            # Check 1: a | b*c => (a/g) | c
            k1 = a // g
            for c in multiples_of[k1]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
            
            # Check 2: b | a*c => (b/g) | c
            k2 = b // g
            for c in multiples_of[k2]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
                    
        return True

    def greedy_construction(order, checker):
        in_set = [False] * 257
        current_set = []
        for a in order:
            if checker(a, in_set):
                in_set[a] = True
                current_set.append(a)
        return current_set

    best_set = []
    
    # --- Strategy 1: Deterministic Greedy with specific orderings ---
    # Order 1: Descending Largest Prime Factor
    order_lpf_desc = sorted(candidates, key=lambda x: (-lpf[x], -x))
    S = greedy_construction(order_lpf_desc, can_add)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 2: Descending Value
    order_desc = sorted(candidates, reverse=True)
    S = greedy_construction(order_desc, can_add)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 3: Ascending Value
    order_asc = sorted(candidates)
    S = greedy_construction(order_asc, can_add)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 4: Primes first
    primes = [x for x in candidates if lpf[x] == x]
    composites = [x for x in candidates if lpf[x] != x]
    order_primes_first = sorted(primes, reverse=True) + sorted(composites, reverse=True)
    S = greedy_construction(order_primes_first, can_add)
    if len(S) > len(best_set):
        best_set = S[:]

    # --- Strategy 2: Large Primes + Antichain Greedy ---
    # We force include all primes > 128.
    # Then we greedily add from candidates_rest using the antichain checker.
    # This is valid because if we include all large primes, the rest MUST be an antichain.
    # And if the rest is an antichain, we still need to check product divisibility.
    # The checker `can_add_antichain` does both.
    
    # Initialize in_set with primes_large
    in_set_la = [False] * 257
    current_set_la = []
    for p in primes_large:
        in_set_la[p] = True
        current_set_la.append(p)
    
    # Now greedily add from candidates_rest
    # Try different orders for candidates_rest
    # Order: Descending LPF
    order_rest_lpf = sorted(candidates_rest, key=lambda x: (-lpf[x], -x))
    for a in order_rest_lpf:
        if can_add_antichain(a, in_set_la):
            in_set_la[a] = True
            current_set_la.append(a)
            
    if len(current_set_la) > len(best_set):
        best_set = current_set_la[:]

    # Order: Descending Value
    in_set_la2 = [False] * 257
    current_set_la2 = []
    for p in primes_large:
        in_set_la2[p] = True
        current_set_la2.append(p)
        
    order_rest_desc = sorted(candidates_rest, reverse=True)
    for a in order_rest_desc:
        if can_add_antichain(a, in_set_la2):
            in_set_la2[a] = True
            current_set_la2.append(a)
            
    if len(current_set_la2) > len(best_set):
        best_set = current_set_la2[:]

    # --- Strategy 3: Randomized Greedy Restarts ---
    import random
    
    num_restarts = 0
    max_restarts = 2000
    
    while time.time() < deadline - 3.0 and num_restarts < max_restarts:
        # Random shuffle of all candidates
        random.shuffle(candidates)
        S = greedy_construction(candidates, can_add)
        num_restarts += 1
        if len(S) > len(best_set):
            best_set = S[:]
            
        # Random shuffle of rest candidates with large primes fixed
        if num_restarts % 2 == 0:
            in_set_rand = [False] * 257
            current_set_rand = []
            for p in primes_large:
                in_set_rand[p] = True
                current_set_rand.append(p)
            
            random.shuffle(candidates_rest)
            for a in candidates_rest:
                if can_add_antichain(a, in_set_rand):
                    in_set_rand[a] = True
                    current_set_rand.append(a)
                    
            if len(current_set_rand) > len(best_set):
                best_set = current_set_rand[:]

    result = [int(x) for x in best_set]
    return result
