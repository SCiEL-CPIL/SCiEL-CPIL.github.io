import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe within 60s
    
    MAX_VAL = 10000
    N = 12
    MAX_SUM = 2 * MAX_VAL

    # Helper to generate S-smooth numbers up to limit
    def gen_smooth(primes, limit):
        nums = {1}
        changed = True
        while changed:
            changed = False
            for p in primes:
                for n in list(nums):
                    prod = n * p
                    if prod <= limit:
                        if prod not in nums:
                            nums.add(prod)
                            changed = True
        return sorted(nums)

    # Helper to check if n is S-smooth (or get its prime factors within S)
    # We need to check if u+v is S-smooth.
    # Faster: Precompute a set of S-smooth numbers up to MAX_SUM.
    
    # Strategy:
    # 1. Try S = {2, 3, 5, 7}
    # 2. If no 12-clique, try S = {2, 3, 5, 7, 11}
    # 3. If no 12-clique, try S = {2, 3, 5, 7, 11, 13}
    # 4. Fallback to local search if all fail (unlikely to fail for small S)
    
    prime_sets = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17]
    ]
    
    best_solution = None
    best_score = float('inf')
    
    for primes in prime_sets:
        if time.time() > deadline - 5:
            break
            
        # Generate smooth numbers up to MAX_VAL for candidates
        # We restrict candidates to smooth numbers themselves to increase density
        candidates = gen_smooth(primes, MAX_VAL)
        
        if len(candidates) < N:
            continue
            
        # Generate smooth sums up to MAX_SUM to check compatibility
        smooth_sums = set(gen_smooth(primes, MAX_SUM))
        
        # Build adjacency
        # We only care about distinct indices, but values must be distinct.
        # Since candidates are unique, indices map 1-to-1 to values.
        
        # Greedy Clique Search with Backtracking
        # Sort candidates by degree (number of compatible neighbors)
        # To save time, we can just pick a subset of candidates.
        # Actually, let's just use the list.
        
        # Precompute adjacency bitsets or lists
        # n_cand might be ~1000. 1000x1000 matrix is 1M entries. Fine.
        n_cand = len(candidates)
        adj = [set() for _ in range(n_cand)]
        
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                s = candidates[i] + candidates[j]
                if s in smooth_sums:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Find a clique of size N
        # Use a simple recursive backtracking
        # Sort by degree descending to guide search
        order = sorted(range(n_cand), key=lambda x: len(adj[x]), reverse=True)
        
        found_clique = None
        
        def find_clique(current_clique, available_indices):
            nonlocal found_clique
            if len(current_clique) == N:
                found_clique = current_clique[:]
                return True
            if not available_indices:
                return False
                
            # Heuristic: Pick the node with highest degree in available_indices
            # Or just iterate through available_indices sorted by degree
            # To speed up, we can limit the branching factor or use a time check
            
            # We want to find ANY clique.
            # Let's iterate through available indices.
            # If we pick a node, the next available must be neighbors of ALL current clique members.
            
            # Optimization: If len(current_clique) + len(available_indices) < N, return False
            
            # We'll just do a standard DFS
            # To avoid deep recursion on dense graphs, we can be smart.
            
            # Sort available_indices by degree in the induced subgraph?
            # Too expensive. Just use global degree order.
            
            for idx in order:
                if idx not in available_indices:
                    continue
                
                # Check if idx connects to all in current_clique
                compatible = True
                for c in current_clique:
                    if idx not in adj[c]:
                        compatible = False
                        break
                        
                if not compatible:
                    continue
                
                # Try adding idx
                current_clique.append(idx)
                
                # New available: intersection of available_indices (excluding idx) and neighbors of idx
                # And neighbors of all current_clique members?
                # Actually, available_indices already represents nodes connected to all previous choices?
                # No, standard Bron-Kerbosch or simple backtracking:
                # If we maintain `available` as nodes that connect to ALL in `current_clique`,
                # then when we pick `idx`, the new available is `available` ∩ `adj[idx]` \ {idx}.
                
                new_available = set()
                for a in available_indices:
                    if a != idx and a in adj[idx]:
                        new_available.add(a)
                        
                if find_clique(current_clique, new_available):
                    return True
                    
                current_clique.pop()
                
                if time.time() > deadline - 2:
                    return False
                    
            return False

        # Initial available: all nodes
        # We can prune initial candidates to top K by degree to make it faster
        # If no clique in top K, try larger K or full set.
        
        # Let's try with a subset first to be fast
        # If the graph is dense, a small subset should work.
        initial_pool = order[:200] 
        initial_available = set(initial_pool)
        
        if not find_clique([], initial_available):
            # Try with full set
            initial_available = set(range(n_cand))
            if not find_clique([], initial_available):
                # Try with a different heuristic or just fail this S
                pass

        if found_clique:
            solution = [candidates[i] for i in found_clique]
            # Verify distinctness (guaranteed by set indices) and bounds
            if len(set(solution)) == N and all(1 <= x <= MAX_VAL for x in solution):
                return solution

    # Fallback: Local Search
    # If the smooth clique approach fails (e.g. no 12-clique for small S),
    # we use a local search to minimize the number of distinct primes.
    
    # Precompute SPF for factorization
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    # Start with a random or structured seed
    import random
    random.seed(42)
    
    # Seed: multiples of 2, 3, 5, 7
    seed_candidates = gen_smooth([2,3,5,7], MAX_VAL)
    if len(seed_candidates) >= N:
        A = random.sample(seed_candidates, N)
    else:
        A = random.sample(range(1, MAX_VAL + 1), N)
        
    # Calculate current score
    def calc_score(A):
        primes = set()
        for i in range(N):
            for j in range(i+1, N):
                primes.update(get_prime_factors(A[i] + A[j]))
        return len(primes)
        
    current_score = calc_score(A)
    best_A = A[:]
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    while time.time() < deadline - 1:
        if T < T_min:
            break
        i = random.randint(0, N - 1)
        new_val = random.randint(1, MAX_VAL)
        
        # Check distinctness
        if any(A[j] == new_val for j in range(N) if j != i):
            continue
            
        # Calculate delta score
        # Remove old pairs involving i, add new pairs
        old_primes = set()
        new_primes = set()
        
        # We need counts to know if a prime disappears completely.
        # This is expensive to maintain perfectly in this quick fallback.
        # Given the smooth clique success is likely, this fallback is just for safety.
        # Let's just recalculate full score for simplicity in fallback, or use a heuristic.
        # Actually, let's just do a full recalc if we do many steps.
        # But 60s is enough for ~1000 full recalc steps (12 choose 2 = 66 sums).
        
        old_val = A[i]
        A[i] = new_val
        new_score = calc_score(A)
        
        if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
        else:
            A[i] = old_val
            
        T *= cooling_rate
        
    return best_A
