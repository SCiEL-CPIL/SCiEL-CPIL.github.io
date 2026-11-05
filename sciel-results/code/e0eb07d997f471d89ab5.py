import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s
    
    N = 256
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build adjacency/conflict sets
    # adj[i] contains j (0-indexed) such that (i+1) + (j+1) is a perfect square
    adj = [set() for _ in range(N)]
    
    # Identify vertices with self-loops (2*v is a square)
    # These vertices cannot be in the solution set.
    forbidden = set()
    
    for i in range(N):
        v = i + 1
        # Check self-loop
        if 2 * v in squares:
            forbidden.add(i)
        
        # Find neighbors
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                adj[i].add(j)
    
    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if i not in forbidden]
    
    # Structural Insight: Modular Partitioning
    # x % 4 == 1 or x % 4 == 3 implies 2x % 4 == 2, so 2x is never a square.
    # Also, if a % 4 == b % 4 == 1, a+b % 4 == 2 (not square).
    # If a % 4 == b % 4 == 3, a+b % 4 == 6 % 4 == 2 (not square).
    # So sets R1 = {x | x%4==1} and R3 = {x | x%4==3} are independent sets.
    # We can use one as a strong seed.
    
    def get_modular_seed(residue):
        return set(i for i in candidates if (i + 1) % 4 == residue)

    best_size = 0
    best_set = frozenset()
    
    def local_improve(s):
        """Try to add vertices that don't conflict. Returns a set."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            for v in candidates:
                if v in s:
                    continue
                if not (adj[v] & s):
                    s.add(v)
                    improved = True
        return s

    def local_search(s, max_iterations=50):
        """
        Local search with 'Remove One, Fill' moves.
        """
        s = set(s)
        s = local_improve(s)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations and (time.time() - start_time < DEADLINE - 1):
            improved = False
            iter_count += 1
            # Try removing each vertex in s
            # To avoid bias, shuffle the order
            vertices_to_try = list(s)
            random.shuffle(vertices_to_try)
            
            for v in vertices_to_try:
                # Remove v
                s_temp = s - {v}
                # Fill greedily
                s_new = local_improve(s_temp)
                
                if len(s_new) > len(s):
                    s = s_new
                    improved = True
                    break # Restart the loop with the new s
                # If no improvement, we don't keep s_temp, we continue trying other v's
                # Note: s remains unchanged in this iteration if no v yielded improvement
            # If we went through all v's and didn't improve, we are stuck in this neighborhood
            # We could try a random perturbation here, but for now, just return
            
        return s

    # Initial seeds based on modular partitioning
    seeds = []
    # Seed 1: Residue 1
    seeds.append(get_modular_seed(1))
    # Seed 2: Residue 3
    seeds.append(get_modular_seed(3))
    # Seed 3: Random
    seeds.append(set(random.sample(candidates, min(50, len(candidates)))))
    
    # Evaluate seeds
    for seed in seeds:
        s = local_search(seed)
        if len(s) > best_size:
            best_size = len(s)
            best_set = frozenset(s)
            
    # Iterated Local Search
    # We will perturb the best solution and try to improve it
    while time.time() - start_time < DEADLINE:
        # Perturbation: Randomly remove a small fraction of vertices from best_set
        # and then refill, or just start from a random subset of best_set?
        # A common ILS move: pick k vertices from best_set, remove them, refill.
        # Or: pick k vertices from best_set, replace with random candidates?
        
        # Let's try: Remove 5 random vertices from best_set, then local_search
        if best_set:
            s_current = set(best_set)
            # Remove a few random vertices to escape local optimum
            num_remove = random.randint(2, 5)
            vertices_to_remove = random.sample(list(s_current), min(num_remove, len(s_current)))
            for v in vertices_to_remove:
                s_current.remove(v)
            
            s = local_search(s_current)
            
            if len(s) > best_size:
                best_size = len(s)
                best_set = frozenset(s)
        
        # Also try a fresh random restart occasionally
        if random.random() < 0.1:
            s = set(random.sample(candidates, min(30, len(candidates))))
            s = local_search(s)
            if len(s) > best_size:
                best_size = len(s)
                best_set = frozenset(s)

    # Convert 0-based indices back to 1-based values
    return sorted([v + 1 for v in best_set])

# EVOLVE-BLOCK-END
