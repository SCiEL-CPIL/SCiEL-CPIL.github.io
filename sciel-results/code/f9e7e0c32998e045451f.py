"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Tighter margin to ensure safety

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]

    primes_up_to_20000 = sieve(20000)

    def get_smooth_numbers(P, limit):
        smooth = set()
        for n in range(2, limit + 1):
            m = n
            for p in P:
                while m % p == 0:
                    m //= p
            if m == 1:
                smooth.add(n)
        return smooth

    def find_clique_fast(adj, vertices, min_size, deadline):
        """Fast clique finder with timeout."""
        if len(vertices) < min_size:
            return None
        
        # Order by degree descending to find cliques faster
        ordered_verts = sorted(vertices, key=lambda v: -len(adj[v]))
        
        def backtrack(current_clique, candidates):
            if time.time() > deadline:
                return None
            if len(current_clique) >= min_size:
                return current_clique[:]
            if not candidates:
                return None
            if len(current_clique) + len(candidates) < min_size:
                return None
            
            for v in candidates:
                new_clique = current_clique + [v]
                # Candidates must be neighbors of v and in the current candidate set
                # To maintain order and avoid duplicates, we filter candidates that come after v in the ordered list?
                # Simpler: just intersect with adj[v]
                new_candidates = [c for c in candidates if c in adj[v]]
                result = backtrack(new_clique, new_candidates)
                if result is not None:
                    return result
            return None

        return backtrack([], ordered_verts)

    best_result = None
    best_score = float('inf')

    # Try prime sets: {2,3}, {2,3,5}, {2,3,5,7}
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7]
    ]
    
    search_limits = [500, 1000, 2000]

    for P in prime_sets:
        if time.time() > deadline - 5:
            break
            
        smooth_nums = get_smooth_numbers(P, 4000) # Sums up to 4000 (2*2000)
        
        for max_val in search_limits:
            if time.time() > deadline - 2:
                break
                
            candidates = list(range(1, max_val + 1))
            adj = {v: set() for v in candidates}
            
            # Build graph
            # Optimization: Only check pairs where sum is in smooth_nums
            # smooth_nums is a set for O(1) lookup
            for i, a in enumerate(candidates):
                for b in candidates[i+1:]:
                    s = a + b
                    if s in smooth_nums:
                        adj[a].add(b)
                        adj[b].add(a)
            
            # Filter candidates that have degree >= 11 (necessary for clique of size 12)
            valid_candidates = [v for v in candidates if len(adj[v]) >= 11]
            
            if len(valid_candidates) < 12:
                continue
                
            # Find clique
            clique = find_clique_fast(adj, valid_candidates, 12, deadline - 1)
            
            if clique is not None:
                result = sorted(clique)
                # Calculate score
                distinct_primes = set()
                for i in range(len(result)):
                    for j in range(i+1, len(result)):
                        s = result[i] + result[j]
                        for p in P:
                            while s % p == 0:
                                distinct_primes.add(p)
                                s //= p
                        if s > 1:
                            distinct_primes.add(s)
                
                score = len(distinct_primes)
                if score < best_score:
                    best_score = score
                    best_result = result
                    # If we achieved the theoretical minimum for this P, break early
                    if score == len(P):
                        return best_result
            else:
                # If no clique found, try larger range or next P
                pass

    if best_result is not None:
        return best_result
        
    # Fallback: Return a set with few primes, even if not minimal
    # Using powers of 2 often results in few distinct primes in sums?
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # Sums: 1+2=3, 1+4=5, ...
    # Let's just return a valid set. The evaluator minimizes value.
    # A random set of 12 numbers will have many primes.
    # A set of multiples of a large number?
    # Let's return the first 12 2,3-smooth numbers, hoping sums are smooth-ish?
    # No, better to return a set that definitely works with a small number of primes.
    # If we pick all numbers to be 1? No, distinct.
    # Let's pick 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.
    # Sums will have primes up to 21.
    
    # Better fallback: Greedy construction minimizing new primes
    # But for now, return a simple valid set to avoid 0 score.
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# EVOLVE-BLOCK-END
