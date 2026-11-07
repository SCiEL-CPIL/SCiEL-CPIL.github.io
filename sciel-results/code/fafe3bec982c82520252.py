import time
import random
from math import gcd

def construct():
    deadline = time.time() + 55.0
    
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
        if lpf[i] == 0:  # i is prime
            for j in range(i, 257, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    candidates = list(range(2, 257))
    
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

    def greedy_construction(order):
        in_set = [False] * 257
        current_set = []
        for a in order:
            if can_add(a, in_set):
                in_set[a] = True
                current_set.append(a)
        return current_set

    def try_improve(S):
        """
        Try to improve set S by removing one element and greedily adding others.
        Returns improved set if found, else original S.
        """
        if time.time() > deadline - 3.0:
            return S
        
        best = S[:]
        best_size = len(S)
        
        # Try removing each element and greedily adding others
        for i in range(len(S)):
            if time.time() > deadline - 3.0:
                break
            x = S[i]
            S_without_x = S[:i] + S[i+1:]
            in_set = [False] * 257
            for y in S_without_x:
                in_set[y] = True
            
            # Try to add elements not in S_without_x
            # Use a random order for diversity
            candidates_to_try = [y for y in range(2, 257) if not in_set[y]]
            random.shuffle(candidates_to_try)
            
            for y in candidates_to_try:
                if can_add(y, in_set):
                    in_set[y] = True
                    S_without_x.append(y)
            
            if len(S_without_x) > best_size:
                best = S_without_x[:]
                best_size = len(S_without_x)
        
        return best

    best_set = []
    
    # --- Strategy 1: Deterministic Greedy with specific orderings ---
    # Order 1: Descending Largest Prime Factor
    order_lpf_desc = sorted(candidates, key=lambda x: (-lpf[x], -x))
    S = greedy_construction(order_lpf_desc)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 2: Descending Value
    order_desc = sorted(candidates, reverse=True)
    S = greedy_construction(order_desc)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 3: Ascending Value
    order_asc = sorted(candidates)
    S = greedy_construction(order_asc)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 4: Primes first
    primes = [x for x in candidates if lpf[x] == x]
    composites = [x for x in candidates if lpf[x] != x]
    order_primes_first = sorted(primes, reverse=True) + sorted(composites, reverse=True)
    S = greedy_construction(order_primes_first)
    if len(S) > len(best_set):
        best_set = S[:]
    
    # --- Strategy 2: Large Primes + Greedy ---
    # Initialize with primes > 128
    primes_large = [p for p in candidates if lpf[p] == p and p > 128]
    candidates_rest = [x for x in candidates if x not in primes_large]
    
    in_set_la = [False] * 257
    current_set_la = []
    for p in primes_large:
        in_set_la[p] = True
        current_set_la.append(p)
    
    # Greedily add from candidates_rest in descending LPF order
    order_rest_lpf = sorted(candidates_rest, key=lambda x: (-lpf[x], -x))
    for a in order_rest_lpf:
        if can_add(a, in_set_la):
            in_set_la[a] = True
            current_set_la.append(a)
            
    if len(current_set_la) > len(best_set):
        best_set = current_set_la[:]
    
    # --- Strategy 3: Local Search on best_set ---
    if len(best_set) > 0:
        improved = try_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved
    
    # --- Strategy 4: Randomized Greedy Restarts with Local Search ---
    num_restarts = 0
    max_restarts = 3000
    
    while time.time() < deadline - 5.0 and num_restarts < max_restarts:
        # Random shuffle of all candidates
        random.shuffle(candidates)
        S = greedy_construction(candidates)
        num_restarts += 1
        
        # Try local search on this set
        S_improved = try_improve(S)
        
        if len(S_improved) > len(best_set):
            best_set = S_improved[:]
    
    # Final polish: try to add any remaining elements
    if len(best_set) > 0:
        in_set = [False] * 257
        for x in best_set:
            in_set[x] = True
        
        # Try adding in descending order
        for a in range(256, 1, -1):
            if not in_set[a]:
                if can_add(a, in_set):
                    in_set[a] = True
                    best_set.append(a)
        
        # Try adding in ascending order
        in_set = [False] * 257
        for x in best_set:
            in_set[x] = True
        for a in range(2, 257):
            if not in_set[a]:
                if can_add(a, in_set):
                    in_set[a] = True
                    best_set.append(a)
    
    result = [int(x) for x in best_set]
    return result
