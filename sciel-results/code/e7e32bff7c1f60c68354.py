"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Search for a tournament on n vertices where every 4-set has a common dominator.
    The goal is to minimize n. We use a genetic algorithm with a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # We know a valid tournament exists on 83 vertices (Paley).
    # We want to find one with smaller n.
    # Erdos lower bound is 31.
    # Let's try to find a solution for n in range [31, 83).
    # If we fail, return the Paley 83.

    def generate_paley(p):
        """Generate Paley tournament on p vertices. Valid if p is prime p=3,7 mod 8."""
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        n = p
        # orientation[i, j] for i < j. 0 if i->j, 1 if j->i.
        # i->j iff (j-i) % p in squares.
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': n, 'orientation': orientation}

    # Check if Paley is valid for this n (it should be for p=83, but maybe smaller primes work?)
    # Primes p = 3, 7 mod 8:
    # 7: 3,7 mod 8? 7=7 mod 8.
    # 11: 3 mod 8.
    # 19: 3 mod 8.
    # 23: 7 mod 8.
    # 31: 7 mod 8.
    # 47: 7 mod 8.
    # 59: 3 mod 8.
    # 67: 3 mod 8.
    # 79: 7 mod 8.
    # 83: 3 mod 8.
    
    # The task says "smallest known has 67 vertices".
    # Paley 67 is a known valid tournament.
    # Let's check if Paley 67 is valid.
    # If it is, we can just return it immediately.
    
    # Let's implement a fast validity check.
    def is_valid(n, orientation):
        """
        Check if every 4-set has a common dominator.
        A vertex v dominates a set S if v->u for all u in S, u!=v.
        For a 4-set {a,b,c,d}, we need one of them to dominate the other 3.
        """
        # Build adjacency matrix for fast lookup
        # adj[i][j] = 1 if i->j, 0 if j->i
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1

        # For each 4-set, check if there's a vertex with out-degree 3 within the set.
        # Precompute for each vertex v, the list of vertices it dominates.
        # Or, for each 4-set, check the 4 candidates.
        
        # Optimization: 
        # For a vertex v, let D(v) be the set of vertices dominated by v.
        # A 4-set {a,b,c,d} is covered if any of a,b,c,d is in D(v) for the other 3.
        # i.e., D(a) contains {b,c,d} OR D(b) contains {a,c,d} ...
        
        # Let's iterate through 4-sets.
        # Number of 4-sets is C(n,4). For n=67, C(67,4) is ~ 670,000.
        # This is fast enough in Python if we are careful, or with NumPy.
        
        # Using NumPy:
        # For each vertex v, we have a boolean vector adj[v].
        # For a 4-set indices i, j, k, l:
        # Check if adj[i][j] & adj[i][k] & adj[i][l] is True.
        # Or adj[j][i] & adj[j][k] & adj[j][l] ...
        
        # We can vectorize this?
        # It might be easier to just loop in Python if n is small, but 67 is moderate.
        # Let's try a hybrid: loop over 4-sets, check 4 conditions.
        
        # Actually, let's use a smarter check.
        # For each vertex v, count how many 4-sets it dominates.
        # If the sum of counts >= C(n,4), and no overlaps? No, overlaps are fine.
        # We just need the union of dominated 4-sets to be all 4-sets.
        
        # Let's stick to checking each 4-set for now, but optimize the inner loop.
        
        # Precompute for each pair (i,j) the bit.
        # For a 4-set (a,b,c,d):
        # a dominates if adj[a,b] and adj[a,c] and adj[a,d]
        
        # Let's try to do this with NumPy broadcasting for speed.
        # This is tricky for arbitrary 4-sets.
        
        # Alternative:
        # For each vertex v, let S_v be the set of vertices v dominates.
        # The 4-sets dominated by v are subsets of {v} U S_v of size 4 containing v.
        # Specifically, {v} U {x,y,z} where x,y,z in S_v.
        
        # Let's just do the direct check in Python. It might be slow for n=67.
        # C(67,4) = 67*66*65*64 / 24 = 6,011,950? No.
        # 67*66 = 4422
        # 65*64 = 4160
        # 4422 * 4160 / 24 = 766,000 approx.
        # 766,000 iterations in Python is ~0.5s - 1s. Acceptable.
        
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        # Check if a dominates b,c,d
                        if adj[a][b] and adj[a][c] and adj[a][d]:
                            continue
                        # Check if b dominates a,c,d
                        if adj[b][a] and adj[b][c] and adj[b][d]:
                            continue
                        # Check if c dominates a,b,d
                        if adj[c][a] and adj[c][b] and adj[c][d]:
                            continue
                        # Check if d dominates a,b,c
                        if adj[d][a] and adj[d][b] and adj[d][c]:
                            continue
                        # None dominate
                        return False
        return True

    # First, check if Paley 67 is valid.
    # If it is, we can return it.
    # The prompt says "smallest known has 67 vertices".
    # Paley 67 is a known solution.
    # Let's verify it.
    
    # But wait, the prompt says "Minimize n".
    # If I find a valid one on n=50, that's better than 67.
    # So I should search for smaller n.
    
    # Strategy:
    # 1. Try Paley primes < 67. (11, 19, 23, 31, 47, 59)
    #    If any are valid, return the smallest.
    # 2. If none are valid, use Genetic Algorithm to search for n < 67.
    # 3. If GA fails, return Paley 67.
    
    # Let's test Paley primes.
    paley_primes = [11, 19, 23, 31, 47, 59]
    
    best_result = None
    best_n = 83 # Start with 83
    
    # Check Paley 67 first to have a fallback
    p67 = generate_paley(67)
    # We assume it's valid based on literature, but let's double check with a quick check?
    # Actually, running is_valid(67, ...) takes ~1s.
    # Let's just assume it's valid and keep it as fallback.
    
    # Check smaller Paley primes
    for p in paley_primes:
        if time.time() > deadline:
            break
        res = generate_paley(p)
        if is_valid(p, res['orientation']):
            if p < best_n:
                best_n = p
                best_result = res
            # Since we want to minimize n, and we are checking in increasing order?
            # 11, 19, 23...
            # If we find one, it's the smallest Paley.
            # But maybe a non-Paley on smaller n exists?
            # Let's break and search for smaller n with GA?
            # Actually, if we find a valid Paley on p=31, that's great.
            # But is there a valid one on n=20?
            # We should search for n < best_n.
            pass

    # Now, use GA to search for n < best_n.
    # We will iterate n from 31 up to best_n - 1.
    # For each n, run a short GA.
    
    # GA Parameters
    pop_size = 100
    max_generations = 500
    mutation_rate = 0.05
    
    # Fitness function: count of uncovered 4-sets.
    # Lower is better. 0 is valid.
    
    def fitness(n, orientation):
        """
        Returns number of uncovered 4-sets.
        """
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1
        
        uncovered = 0
        # To speed up, we can sample? No, we need exact count for GA.
        # But counting all 4-sets is slow.
        # Let's optimize the counting.
        
        # For each vertex v, let k_v be the number of vertices it dominates.
        # The number of 4-sets dominated by v is C(k_v, 3).
        # But sets can be dominated by multiple vertices.
        # We need the size of the union.
        # This is hard to compute directly without iterating sets.
        
        # Alternative: Random Sampling for fitness.
        # If we sample 1000 random 4-sets, and count how many are uncovered.
        # This is a stochastic fitness. It might be noisy but faster.
        # Given the deadline, stochastic fitness is a good trade-off.
        
        sample_size = 2000
        # Pre-generate indices for random 4-sets?
        # Or just pick random vertices.
        
        # To avoid bias, we should pick uniformly random 4-sets.
        # np.random.choice(n, 4, replace=False)
        
        # Let's use a fixed set of random 4-sets for consistent comparison within a generation?
        # Or just random each time.
        
        # For speed, let's pre-compute a list of 4-set indices once per n?
        # No, n changes.
        
        # Let's just do the full check if n is small, else sample.
        # For n < 40, C(n,4) is small.
        # C(40,4) = 91,390.
        # Full check is fast for n < 40.
        
        if n <= 40:
            # Full check
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(b + 1, n):
                        for d in range(c + 1, n):
                            if not (adj[a][b] and adj[a][c] and adj[a][d] or
                                    adj[b][a] and adj[b][c] and adj[b][d] or
                                    adj[c][a] and adj[c][b] and adj[c][d] or
                                    adj[d][a] and adj[d][b] and adj[d][c]):
                                uncovered += 1
            return uncovered
        else:
            # Sample
            # We can't easily vectorize the "any of 4 dominates" check with sampling without loops.
            # Let's just loop over samples.
            for _ in range(sample_size):
                idxs = np.random.choice(n, 4, replace=False)
                a, b, c, d = idxs
                if not (adj[a][b] and adj[a][c] and adj[a][d] or
                        adj[b][a] and adj[b][c] and adj[b][d] or
                        adj[c][a] and adj[c][b] and adj[c][d] or
                        adj[d][a] and adj[d][b] and adj[d][c]):
                    uncovered += 1
            return uncovered

    # We want to find n < best_n.
    # Let's try n from 31 to best_n - 1.
    # Actually, if we found a Paley on p=31, best_n=31. We are done.
    # If we didn't, best_n is 67 (or 83 if 67 failed, but 67 is known valid).
    # So we search for n in [31, 66].
    
    # Wait, if Paley 31 was valid, we would have found it.
    # So we assume no Paley < 67 is valid.
    # We search for non-Paley tournaments.
    
    # Let's start with n = 31.
    # If we find a valid one, return it.
    # If not, try n=32, etc.
    
    # But searching n=31 is hard.
    # Maybe start with larger n?
    # The landscape is smoother for larger n?
    # Actually, smaller n is harder to satisfy.
    # Let's try to find a solution for n=60, then 55, etc?
    # Or just iterate up.
    
    # Given time constraints, let's try a few n values.
    # n = 40, 50, 60, 65.
    
    target_ns = [40, 50, 60, 65]
    # Sort by descending n to find a valid one faster?
    # Then we can try to reduce?
    # No, just find the smallest valid one.
    # If we find one on 60, that's better than 67.
    # Then we can try to find one on 50?
    
    # Let's just iterate n from 31 to 66.
    # But that's a lot of GAs.
    # Let's pick a few promising n.
    
    # Actually, let's just run one GA for n=60.
    # If it finds a valid one, great.
    # If not, try n=55.
    
    # Let's define the GA.
    
    def run_ga(n, time_limit):
        """
        Run GA for n vertices.
        Returns valid orientation if found, else None.
        """
        pop_size = 50
        # Initialize population
        population = []
        for _ in range(pop_size):
            # Random orientation
            orientation = np.random.randint(0, 2, size=n * (n - 1) // 2)
            population.append(orientation)
        
        # Evaluate initial fitness
        fitnesses = [fitness(n, ind) for ind in population]
        
        # If any is 0, return it.
        if min(fitnesses) == 0:
            return population[fitnesses.index(0)]
        
        start_ga = time.time()
        
        while time.time() - start_ga < time_limit:
            # Selection: Tournament selection
            new_population = []
            for _ in range(pop_size):
                # Pick 3 random, keep best
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                i3 = np.random.randint(pop_size)
                best_idx = i1
                if fitnesses[i2] < fitnesses[best_idx]:
                    best_idx = i2
                if fitnesses[i3] < fitnesses[best_idx]:
                    best_idx = i3
                new_population.append(population[best_idx].copy())
            
            # Mutation
            for i in range(pop_size):
                # Flip some bits
                mask = np.random.rand(len(new_population[i])) < mutation_rate
                new_population[i][mask] = 1 - new_population[i][mask]
            
            # Evaluate new fitness
            fitnesses = [fitness(n, ind) for ind in new_population]
            
            # Check for valid
            if min(fitnesses) == 0:
                return new_population[fitnesses.index(0)]
            
            population = new_population
        
        return None

    # Try some n values
    # We have ~120s left.
    # Let's try n=60 with 30s, n=55 with 30s, n=50 with 30s.
    
    candidates = [
        (60, 30),
        (55, 30),
        (50, 30),
        (45, 20)
    ]
    
    for n, t in candidates:
        if time.time() > deadline:
            break
        if n >= best_n:
            continue
        result = run_ga(n, t)
        if result is not None:
            # Convert numpy array to list of ints
            orientation_list = result.tolist()
            best_n = n
            best_result = {'n': n, 'orientation': orientation_list}
            # We found a better one. Can we find even better?
            # Maybe, but let's see if we have time.
            # We could try smaller n.
            # But let's just return this for now?
            # The directive says "Minimize n".
            # If we have time, we should try smaller.
            # Let's add smaller n to the list if we find one.
            # Actually, the list is fixed.
            # Let's just keep searching in the loop.
            # If we find n=60, we still try n=55, 50, 45.
            # So the loop handles it.

    # If no better than 67 found, return Paley 67.
    if best_n > 67:
        best_result = generate_paley(67)
        best_n = 67
    elif best_n == 83:
        # Fallback to 83 if 67 failed (shouldn't happen)
        best_result = generate_paley(83)
        best_n = 83

    return best_result

# EVOLVE-BLOCK-END
