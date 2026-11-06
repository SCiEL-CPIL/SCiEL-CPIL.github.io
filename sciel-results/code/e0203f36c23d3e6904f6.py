import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MAX_VAL = 20000
    # Sieve of Eratosthenes for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def is_smooth(n, max_prime):
        if n <= 1:
            return True
        while n > 1:
            p = spf[n]
            if p > max_prime:
                return False
            while n % p == 0:
                n //= p
        return True

    def build_adj(N, max_prime):
        """Build adjacency list for graph where edge (u,v) exists if u+v is max_prime-smooth."""
        adj = [set() for _ in range(N + 1)]
        for u in range(1, N + 1):
            for v in range(u + 1, N + 1):
                if is_smooth(u + v, max_prime):
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def find_clique_size_12(adj, N):
        """
        Find a clique of size 12 in the graph defined by adj[1..N].
        Uses a modified Bron-Kerbosch with pivoting and pruning.
        """
        # Pre-filter: nodes must have degree >= 11
        candidates = [i for i in range(1, N + 1) if len(adj[i]) >= 11]
        
        # Sort candidates by degree descending to improve pruning
        candidates.sort(key=lambda x: -len(adj[x]))
        
        # If not enough candidates, return None
        if len(candidates) < 12:
            return None

        # Recursive search
        # To avoid recursion depth issues and speed up, we use a stack-based approach or limited recursion
        # Given N=500, recursion depth 12 is safe.
        
        def expand(clique, cand):
            if len(clique) == 12:
                return clique
            if not cand:
                return None
            
            # Pruning: if we can't reach 12 even with all remaining candidates
            if len(clique) + len(cand) < 12:
                return None

            # Choose a pivot to reduce branching
            # Pivot u should maximize |cand \ N(u)|? No, standard BK uses pivot to minimize branching.
            # We want to pick a pivot v from cand such that we iterate over cand \ N(v).
            # This ensures every maximal clique containing 'clique' is considered.
            # For finding *any* clique of size K, we can be less strict.
            
            # Simple strategy: Pick node with highest degree in cand as next candidate?
            # Or just iterate through cand.
            
            # To speed up, let's pick the node with highest degree in cand first.
            # If it fails, backtrack.
            
            # Optimized: Order candidates by degree in cand
            # This is expensive if done at every level. 
            # Let's just iterate.
            
            for u in cand:
                # Try adding u
                new_clique = clique + [u]
                # New candidates must be neighbors of u
                new_cand = [v for v in cand if v in adj[u]]
                
                res = expand(new_clique, new_cand)
                if res is not None:
                    return res
                    
            return None

        return expand([], candidates)

    # Try increasing sets of primes
    # Prime sets to try
    prime_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    for primes in prime_sets:
        max_p = max(primes)
        # Determine N
        # For small primes, density is low, need larger N.
        # For P={2,3}, N=1000 is reasonable.
        # For P={2,3,5}, N=500 might be enough.
        
        N = 1000 if max_p <= 3 else 500
        
        # Check time
        if time.time() > deadline - 5:
            break
            
        adj = build_adj(N, max_p)
        
        # Check if time is still okay
        if time.time() > deadline - 2:
            continue
            
        clique = find_clique_size_12(adj, N)
        
        if clique:
            # Verify
            # Ensure distinct and <= 10000
            if len(set(clique)) == 12 and max(clique) <= 10000:
                return clique
                
    # Fallback: Local search
    # If clique search fails, use local search to minimize prime count
    def factorize(n):
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
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)

    # Start with a decent initial guess
    # Multiples of 6? Sums are multiples of 6.
    # Primes of 6k are {2,3} + primes of k.
    # If we pick k such that k are powers of 2 or 3?
    # Let's just use the previous best known or random.
    
    random.seed(42)
    best = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    best_score = evaluate(best)
    
    # Local search
    candidate_pool = list(set(random.randint(1, 10000) for _ in range(2000)))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            old_val = best[i]
            for new_val in candidate_pool:
                if new_val == old_val or new_val in best:
                    continue
                candidate = best.copy()
                candidate[i] = new_val
                score = evaluate(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    break
                    
    return best
