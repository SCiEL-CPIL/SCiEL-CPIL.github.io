"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    N = 256
    
    # 1. Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # 1 has no prime factors
    
    def is_compatible(a, b):
        if a == b:
            return False
        return not (prime_factors[a] & prime_factors[b])
    
    # 2. Build conflict graph
    # vertices are 1..256
    # edge between u, v if gcd(u,v) > 1
    conflict = [set() for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if gcd(u, v) > 1:
                conflict[u].add(v)
                conflict[v].add(u)
    
    # 3. Maximum Weight Independent Set via Branch and Bound
    # We want to select a subset S of {1..256} such that no two elements in S share a prime factor,
    # maximizing sum(S).
    
    # Greedy initial solution: sort by value descending, add if compatible with current set
    def greedy_solution():
        selected = set()
        total = 0
        # Try multiple orderings
        best_set = set()
        best_sum = 0
        
        # Order 1: descending value
        for v in sorted(range(1, N + 1), key=lambda x: -x):
            if all(is_compatible(v, u) for u in selected):
                selected.add(v)
                total += v
        if total > best_sum:
            best_sum = total
            best_set = set(selected)
        
        # Order 2: ascending value (might allow more high-value later? unlikely but try)
        selected = set()
        total = 0
        for v in sorted(range(1, N + 1)):
            if all(is_compatible(v, u) for u in selected):
                selected.add(v)
                total += v
        if total > best_sum:
            best_sum = total
            best_set = set(selected)
        
        # Order 3: random
        for _ in range(20):
            selected = set()
            total = 0
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for v in candidates:
                if all(is_compatible(v, u) for u in selected):
                    selected.add(v)
                    total += v
            if total > best_sum:
                best_sum = total
                best_set = set(selected)
        
        return best_set, best_sum
    
    best_set, best_sum = greedy_solution()
    
    # Branch and Bound with tight LP-relaxation upper bound
    # The key insight: for each prime p, we can pick at most one number divisible by p.
    # The LP relaxation upper bound is computed via a greedy fractional packing:
    # - For each prime p, let M_p = {v in candidates : p | v}
    # - We want to maximize sum(v * x_v) subject to for each p, sum_{v in M_p} x_v <= 1, x_v in [0,1]
    # - A valid upper bound: process primes in order, for each prime p not yet "covered", 
    #   take the max-value candidate divisible by p, add its value, and remove it from consideration.
    #   This is NOT tight in general but is a valid upper bound if we're careful.
    # 
    # Actually, a simpler valid upper bound: 
    #   For each candidate v, it can only be selected if for every prime p|v, no other selected 
    #   number is divisible by p. 
    #   A crude but valid bound: sum of all candidates (loose).
    #   Better: For each prime p <= N, define group G_p = {v in candidates : p|v}. 
    #   We can pick at most one from each G_p. But groups overlap.
    #   A valid upper bound is: for each prime p, take max value in G_p, but subtract overlaps.
    #   This is complex. Let's use a different approach.
    #
    # Tighter bound via "dominance": 
    #   If v1 and v2 share a prime factor, we can pick at most one.
    #   So the maximum sum is at most: for each connected component of the conflict graph 
    #   restricted to candidates, take the max weight vertex? No, that's not right either.
    #
    # Let's use a practical bound: 
    #   bound = 0
    #   remaining = set(candidates)
    #   while remaining:
    #       v = max(remaining)  # pick largest
    #       bound += v
    #       # remove all that conflict with v
    #       remaining = {u in remaining : u != v and is_compatible(v, u)}
    #   This is actually the greedy solution value, which is a LOWER bound, not upper.
    #
    # For a valid UPPER bound, we need something >= the true optimum.
    # The sum of all candidates is always a valid upper bound.
    # To make it tighter, we can use:
    #   For each prime p, let S_p = sum of values of candidates divisible by p.
    #   The true optimum is <= min over all "packings" ... this is getting complex.
    #
    # Practical approach: Use the sum bound but with aggressive pruning via 
    #   "include/exclude" ordering and early termination.
    #   Also, use a better branching variable: pick the vertex with highest "impact" 
    #   (most conflicts) to branch on, not just the max value.
    
    def compute_upper_bound(candidates):
        """Compute a valid upper bound on the additional sum from candidates."""
        if not candidates:
            return 0
        # Simple valid upper bound: sum of all candidates
        # Tighter: for each prime p, we can only pick one multiple.
        # The sum of max-value-per-prime-group is NOT a valid upper bound due to overlaps.
        # But we can use: for each candidate v, its "effective weight" is reduced if it 
        # conflicts with high-value candidates.
        # 
        # Let's use a valid bound: 
        #   UB = sum(candidates)  # always valid
        #   Then tighten: for each prime p, if there are k candidates divisible by p,
        #   we can pick at most 1, so we can subtract (k-1) * min_value_in_group? No.
        #
        # Actually, a clean valid upper bound:
        #   Group candidates by their smallest prime factor (or just by each prime they have).
        #   For each prime p, let M_p = max value among candidates divisible by p.
        #   Then UB <= sum over distinct primes p of M_p? No, that's not valid either 
        #   because a number can be divisible by multiple primes.
        #
        # Let's just use sum(candidates) and rely on good branching order.
        return sum(candidates)
    
    # Better: Use a "coloring" based bound or a simple iterative relaxation.
    # For now, let's use sum() but with much better branching strategy.
    
    # Precompute: for each vertex, its conflict set within 1..N
    # conflict[v] is already computed
    
    # Branch and Bound with improved strategy:
    # - Branch on the vertex with highest value (to find good solutions early)
    # - Use sum-based pruning
    # - Order: try include first (to find good solutions quickly)
    
    def branch_and_bound(candidates, current_sum):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(candidates | set())  # current_set is implicit
            return
        
        # Pruning: if current_sum + sum(candidates) <= best_sum, prune
        if current_sum + sum(candidates) <= best_sum:
            return
        
        # Pick branching vertex: highest value
        v = max(candidates)
        
        # Option 1: Include v
        new_candidates = set()
        for u in candidates:
            if u != v and is_compatible(v, u):
                new_candidates.add(u)
        
        # Save current best before exploring
        branch_and_bound(new_candidates, current_sum + v)
        
        # Option 2: Exclude v
        candidates.remove(v)
        branch_and_bound(candidates, current_sum)
        candidates.add(v)
    
    # We need to track the current set to return it. Let's refactor.
    
    def bb_search(candidates, current_set, current_sum):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
        
        # Pruning
        if current_sum + sum(candidates) <= best_sum:
            return
        
        # Pick branching vertex: highest value
        v = max(candidates)
        
        # Option 1: Include v
        new_candidates = set()
        for u in candidates:
            if u != v and is_compatible(v, u):
                new_candidates.add(u)
        
        current_set.add(v)
        bb_search(new_candidates, current_set, current_sum + v)
        current_set.remove(v)
        
        # Option 2: Exclude v
        candidates.remove(v)
        bb_search(candidates, current_set, current_sum)
        candidates.add(v)
    
    # Initial candidates: all numbers 1..N
    initial_candidates = set(range(1, N + 1))
    current_set = set()
    
    bb_search(initial_candidates, current_set, 0)
    
    # Ensure types are correct
    result = list(best_set)
    result.sort()
    return result

# EVOLVE-BLOCK-END
