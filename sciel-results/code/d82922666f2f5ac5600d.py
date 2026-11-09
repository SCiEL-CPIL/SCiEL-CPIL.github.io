import random
import time
import numpy as np

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, primes_list):
    n = len(points)
    if n == 0:
        return True
    for p in primes_list:
        if p > n:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True

def _min_degree_greedy(primes_list, rng, deadline):
    """
    Greedy construction: start with all integers in [0, N], 
    iteratively remove the vertex with minimum conflicts.
    A vertex v has a conflict with u if (v-u) % p == 0 for some small prime p.
    We use a threshold K on primes to keep the graph manageable.
    """
    K = 97  # threshold prime for conflict graph
    
    # Precompute conflict neighbors using bitmasks
    # For each vertex v, compute which other vertices conflict with it
    # Two vertices conflict if their difference is divisible by some prime <= K
    
    # We'll work with numpy arrays for speed
    # Start with all N+1 vertices
    active = np.ones(N + 1, dtype=bool)
    
    # Precompute for each small prime p, the residue classes
    # For a given p, vertices in the same residue class mod p "conflict"
    # (removing one from a full residue class helps admissibility)
    
    # Actually, let's think differently:
    # We want the largest subset A such that for each prime p <= |A|,
    # A does NOT cover all residues mod p.
    
    # Greedy approach: start with all numbers, then remove vertices
    # to "break" full residue classes for small primes.
    
    # Better: use the min-degree idea where degree = number of conflicts
    # A vertex v conflicts with u if they are in the same residue class
    # for some prime p <= K.
    
    # Let's precompute conflict counts using residue class membership
    # For each prime p <= K, group vertices by v % p
    
    small_primes = [p for p in primes_list if p <= K]
    
    # For each vertex, count how many other active vertices share a residue
    # for at least one small prime (this is an approximation of degree)
    
    # Actually, let's use a simpler but effective heuristic:
    # Iteratively find the prime p where we're "most covered" 
    # (all or nearly all residues present) and remove vertices from
    # the least-populated residue class.
    
    points = list(range(N + 1))
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        n = len(points)
        if n == 0:
            break
        
        # Check if admissible
        admissible = True
        for p in small_primes:
            if p > n:
                break
            residues = set()
            for v in points:
                residues.add(v % p)
                if len(residues) == p:
                    admissible = False
                    break
            if not admissible:
                break
        
        if admissible:
            break
        
        # Find which prime is violated and fix it
        for p in small_primes:
            if p > n:
                break
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            
            if len(classes) == p:
                # All residues covered - need to remove some
                # Remove from the smallest class (min degree removal)
                min_size = min(len(v) for v in classes.values())
                candidates = [v for v in classes.values() if len(v) == min_size]
                to_remove = rng.choice(candidates)
                to_remove_set = set(to_remove)
                points = [v for v in points if v not in to_remove_set]
                break
        else:
            # No small prime violated, check larger primes
            for p in primes_list:
                if p > n:
                    break
                if p <= K:
                    continue
                residues = set()
                for v in points:
                    residues.add(v % p)
                    if len(residues) == p:
                        # Violation
                        classes = {}
                        for v in points:
                            r = v % p
                            if r not in classes:
                                classes[r] = []
                            classes[r].append(v)
                        min_size = min(len(v) for v in classes.values())
                        candidates = [v for v in classes.values() if len(v) == min_size]
                        to_remove = rng.choice(candidates)
                        to_remove_set = set(to_remove)
                        points = [v for v in points if v not in to_remove_set]
                        break
            else:
                break
    
    return points

def _polish(points, primes_list, rng, deadline):
    """Local search polish: try to add back removed vertices if possible."""
    points_set = set(points)
    available = [v for v in range(N + 1) if v not in points_set]
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        n = len(points)
        improved = False
        
        # Try adding a random available vertex
        if not available:
            break
        
        v = rng.choice(available)
        test_points = points + [v]
        test_n = len(test_points)
        
        # Check admissibility
        admissible = True
        for p in primes_list:
            if p > test_n:
                break
            residues = set()
            for x in test_points:
                residues.add(x % p)
                if len(residues) == p:
                    admissible = False
                    break
            if not admissible:
                break
        
        if admissible:
            points = test_points
            points_set.add(v)
            available.remove(v)
            improved = True
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N)
    
    best_result = []
    best_len = 0
    
    # Run multiple randomized restarts
    num_runs = 200
    
    for i in range(num_runs):
        if time.time() > deadline:
            break
        
        rng = random.Random(i * 7919 + 13)
        
        # Different initial strategies
        if i % 3 == 0:
            # Start with all, greedily remove
            result = _min_degree_greedy(primes, rng, deadline)
        elif i % 3 == 1:
            # Start with 1 and large primes, then add more
            initial = [1] + [p for p in primes if p * p > N]
            result = _min_degree_greedy(primes, rng, deadline)
            # Polish
            result = _polish(result, primes, rng, deadline)
        else:
            # Start with evens and odds mix
            initial = list(range(0, N + 1, 2)) + [1]
            result = _min_degree_greedy(primes, rng, deadline)
            result = _polish(result, primes, rng, deadline)
        
        if len(result) > best_len:
            best_len = len(result)
            best_result = result
    
    return best_result
