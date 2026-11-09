"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import random

def _is_smooth(n, primes):
    """Check if n is composed only of the given primes."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def _get_prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the union of
    prime factors of all pairwise sums (distinct indices) is minimized.
    
    Strategy: Restrict the universe to numbers that are smooth (only have
    prime factors from a small set). If we pick numbers that are all
    multiples of 2, their sums are even, but may have other factors.
    
    A better approach: Use numbers that are powers of 2, or more generally,
    numbers whose pairwise sums have few distinct prime factors.
    
    Let's try to find a set where all pairwise sums are 7-smooth (only
    prime factors 2, 3, 5, 7). This would give at most 4 distinct primes.
    
    We precompute a compatibility graph where two numbers are connected if
    their sum is 7-smooth, then find a clique of size 12.
    """
    primes_7smooth = [2, 3, 5, 7]
    
    # Precompute which numbers <= 10000 have pairwise sums that are 7-smooth
    # Actually, we need to check for each pair (a, b) if a+b is 7-smooth.
    # Let's build the graph.
    
    max_val = 10000
    n_needed = 12
    
    # For efficiency, let's consider a smaller candidate set.
    # Numbers that are themselves 7-smooth might be good candidates,
    # but their sums might not be 7-smooth.
    
    # Let's just use all numbers 1..10000 and build the graph.
    # But that's 10000 nodes, which is too many for a naive clique search.
    
    # Let's restrict to numbers that are "likely" to have 7-smooth sums with
    # many other numbers. A good heuristic: numbers that are themselves
    # 7-smooth.
    
    candidates = []
    for i in range(1, max_val + 1):
        if _is_smooth(i, primes_7smooth):
            candidates.append(i)
    
    # If we have enough candidates, build the compatibility graph
    if len(candidates) >= n_needed:
        # Build adjacency: two numbers are compatible if their sum is 7-smooth
        n_cand = len(candidates)
        # Use a set of compatible indices for each node
        adj = [set() for _ in range(n_cand)]
        
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                s = candidates[i] + candidates[j]
                if _is_smooth(s, primes_7smooth):
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Find a clique of size n_needed using backtracking
        def find_clique(size):
            # Order candidates by degree (descending) for better pruning
            order = sorted(range(n_cand), key=lambda x: len(adj[x]), reverse=True)
            
            # Map original indices to new order
            # We'll work with the ordered list
            def backtrack(clique, candidates_set):
                if len(clique) == size:
                    return clique
                if not candidates_set:
                    return None
                # Pick the first candidate
                # For better performance, pick the one with highest degree in candidates_set
                # But for simplicity, just iterate
                for v in list(candidates_set):
                    # Check if v is connected to all in clique
                    if all(v in adj[u] for u in clique):
                        new_clique = clique + [v]
                        new_candidates = candidates_set - {v}
                        # Forward checking: only keep candidates connected to v
                        new_candidates = {c for c in new_candidates if v in adj[c]}
                        result = backtrack(new_clique, new_candidates)
                        if result is not None:
                            return result
                return None
            
            # Use a random subset if the full set is too large
            # For now, try with the ordered candidates
            initial_candidates = set(order)
            result = backtrack([], initial_candidates)
            if result is not None:
                return [candidates[i] for i in result]
        
        result = find_clique(n_needed)
        if result is not None:
            # Verify the solution
            all_primes = set()
            for a, b in itertools.combinations(result, 2):
                all_primes |= _get_prime_factors(a + b)
            return result
    
    # Fallback: use a simple heuristic
    # Pick numbers that are multiples of a large power of 2
    # e.g., 2^k for k=0..11, but they must be <= 10000
    # 2^0=1, 2^1=2, ..., 2^13=8192, 2^14=16384 > 10000
    # So we can use 2^0 to 2^13, that's 14 numbers
    # Their sums are 2^i + 2^j = 2^min(i,j) * (1 + 2^|i-j|)
    # The factor (1 + 2^k) may have various prime factors
    
    # Let's try a different fallback: use numbers from a small set
    # that we know work well
    # For example, all even numbers in a small range
    # But let's just return a reasonable set
    
    # Use the first 12 numbers that are 7-smooth
    if len(candidates) >= n_needed:
        return candidates[:n_needed]
    
    # Ultimate fallback
    return list(range(1, 13))

# EVOLVE-BLOCK-END
