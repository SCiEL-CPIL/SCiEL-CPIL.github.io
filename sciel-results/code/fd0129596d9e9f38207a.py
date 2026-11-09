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
    
    # Branch and Bound
    # State: (candidate_set, current_set, current_sum)
    # candidate_set: set of numbers that can still be added (compatible with current_set)
    
    # Upper bound heuristic: sum of all remaining candidates
    # This is a loose bound but fast.
    
    # Better bound: For each prime p, we can only pick one number from {n in candidates : p|n}.
    # This is a packing problem. A simple bound is the sum of max value for each prime group, but groups overlap.
    # Let's use a simple greedy bound: sort candidates by value desc, greedily pick if compatible.
    
    def greedy_bound(candidates):
        """Return an upper bound on the sum we can get from candidates, assuming they are added to an empty set."""
        # This is actually the exact value of the greedy algorithm on candidates, which is a lower bound on the optimum.
        # For an upper bound, we can use the sum of all candidates (very loose).
        # Or, we can use the fact that for each prime p, we pick at most one multiple.
        # Let's try a slightly tighter bound:
        # For each prime p <= N, let M_p be the set of candidates divisible by p.
        # We can pick at most one from each M_p.
        # But a number can be in multiple M_p.
        # This is still complex.
        
        # Let's just use sum(candidates) as a very loose bound.
        # Or better, use the result of a fast greedy on candidates.
        # If greedy(candidates) > best_sum, we explore.
        # This is a valid pruning strategy: if the best we can *greedily* do is worse than current best, 
        # it doesn't mean the optimal is worse, but it's a heuristic.
        # To be safe for correctness, we need a valid upper bound.
        # Sum of all candidates is a valid upper bound.
        
        # Let's try a slightly better valid upper bound:
        # For each candidate v, its value is v.
        # If we group by smallest prime factor, we can pick at most one from each group? No.
        
        # Let's stick with sum(candidates) for now, and rely on the structure to prune.
        # Actually, we can prune if current_sum + sum(candidates) <= best_sum.
        
        return sum(candidates)
    
    # Recursive B&B
    # To avoid deep recursion and overhead, we might use iterative or optimized recursion.
    # N=256 is small enough for recursion if pruning is good.
    
    # Sort candidates initially by value descending?
    # In B&B, the order of branching matters.
    
    def branch_and_bound(candidates, current_set, current_sum):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
        
        # Pruning: if current_sum + sum(candidates) <= best_sum, prune
        if current_sum + sum(candidates) <= best_sum:
            return
        
        # Heuristic: try adding the largest candidate first
        # Pick a candidate to branch on
        # Let's pick the largest value candidate
        v = max(candidates)
        
        # Option 1: Include v
        new_candidates = set()
        for u in candidates:
            if u != v and is_compatible(v, u):
                new_candidates.add(u)
        
        current_set.add(v)
        branch_and_bound(new_candidates, current_set, current_sum + v)
        current_set.remove(v)
        
        # Option 2: Exclude v
        candidates.remove(v)
        branch_and_bound(candidates, current_set, current_sum)
        candidates.add(v)
    
    # Initial candidates: all numbers 1..N
    # We can pre-filter: if a number is "dominated" by another?
    # If a < b and prime_factors(a) is a superset of prime_factors(b) (i.e., b conflicts with more or same),
    # then a is never better than b.
    # Specifically, if b is a multiple of a, then any set containing a can be improved by replacing a with b?
    # Not necessarily, because b might conflict with other elements that a doesn't.
    # But if b is a prime and a is composite, b is usually better.
    
    # Let's just run B&B on all 256.
    # To make it faster, we can order the candidates in the initial call.
    # The B&B above picks max(candidates) each time.
    
    # Optimization: Precompute compatibility matrix
    # comp[u][v] = True if u and v are compatible
    
    # Let's restart with a cleaner B&B implementation using bitmasks or sets efficiently.
    
    # Given the small N, let's try to optimize the B&B loop.
    
    # We'll use a list for candidates to avoid set overhead in sum() and max() if possible,
    # but sets are needed for fast membership and filtering.
    
    # Let's run the B&B
    initial_candidates = set(range(1, N + 1))
    current_set = set()
    
    # Sort initial candidates by value descending to help the "max" pick be efficient?
    # max() on a set is O(N).
    
    branch_and_bound(initial_candidates, current_set, 0)
    
    # Final check
    if time.time() > deadline:
        # Return best found so far
        pass
    
    # Ensure types are correct
    result = list(best_set)
    result.sort() # Just for consistency, order doesn't matter for sum
    return result

# EVOLVE-BLOCK-END
