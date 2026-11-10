import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy: Genetic Algorithm (GA) with a population of 21-vertex graphs.
    Fitness: Negative count of regular induced subgraphs of size 6.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target n=21 to beat the seed. If that fails, fall back to n=20, etc.
    # The task says "Maximize n". A 21-vertex graph would show G(6) >= 22.
    # Known: 20 vertices exist.
    
    def count_conflicts_size6(n, adj):
        """
        Counts the number of regular induced subgraphs of size 6.
        adj is a list of int bitmasks.
        """
        count = 0
        # Iterate over all subsets of size 6
        # For n=21, C(21,6) = 54264. This is fast enough for a GA fitness eval
        # if optimized.
        for S in combinations(range(n), 6):
            mask = 0
            for v in S:
                mask |= (1 << v)
            
            # Check if induced subgraph on mask is regular
            # Degrees in S: deg(v) = popcount(adj[v] & mask)
            # Since |S|=6, possible degrees are 0..5.
            # Regular means all degrees are equal.
            
            # Optimization: Compute degrees for all 6 vertices
            degs = []
            for v in S:
                degs.append((adj[v] & mask).bit_count())
            
            # Check if all equal
            if len(set(degs)) == 1:
                count += 1
                
        return count

    def verify_full(n, adj):
        """
        Checks if the graph has NO regular induced subgraph of size >= 6.
        Returns True if valid.
        """
        # First check size 6 (fastest, most numerous)
        if count_conflicts_size6(n, adj) > 0:
            return False
            
        # Check sizes 7 to n
        for k in range(7, n + 1):
            for S in combinations(range(n), k):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                first_deg = -1
                is_reg = True
                for v in S:
                    deg = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return False
        return True

    def adj_to_edges(n, adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def ga_search(n, pop_size=50, max_generations=100):
        """
        Genetic Algorithm to find a graph on n vertices with no regular induced subgraph of size 6.
        """
        # Initialize population
        population = []
        for _ in range(pop_size):
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            population.append(adj)
        
        best_adj = None
        best_fitness = float('inf') # We want to minimize conflicts (0 is best)
        
        # Pre-compute combinations for size 6 for speed? 
        # combinations(range(n), 6) is generated each time. 
        # For n=21, 54264 combos. 50 pop * 54264 * 6 ops ~ 16M ops per gen.
        # Python is slow. 16M simple ops might take ~1-2s. 
        # 100 gens = 100-200s. Too slow for 110s budget.
        # We need to optimize or reduce pop/gen.
        
        # Let's try smaller pop and fewer gens, or optimize the fitness function.
        # Optimized fitness:
        
        def fitness(adj):
            return count_conflicts_size6(n, adj)
            
        # Let's run GA
        current_time = time.time()
        
        # Estimate time per generation
        # Try 1 generation to measure
        t0 = time.time()
        f0 = fitness(population[0])
        t1 = time.time()
        time_per_eval = (t1 - t0)
        time_per_gen = time_per_eval * pop_size
        
        # Adjust pop_size and generations to fit budget
        remaining_time = deadline - time.time()
        if time_per_gen > 0:
            max_gens = int(remaining_time / time_per_gen)
            max_gens = max(1, min(max_gens, 20)) # Cap at 20 to avoid overfitting to slow path
        else:
            max_gens = 20
            
        # If pop_size is too large, reduce it
        if time_per_gen * 5 > remaining_time:
             pop_size = max(5, pop_size // 4)
             # Re-init population with smaller size? Or just use first pop_size.
             # Let's just slice population.
             population = population[:pop_size]
             
        for gen in range(max_gens):
            if time.time() > deadline:
                break
                
            # Evaluate fitness
            fitnesses = [fitness(adj) for adj in population]
            
            # Update best
            min_idx = np.argmin(fitnesses)
            if fitnesses[min_idx] < best_fitness:
                best_fitness = fitnesses[min_idx]
                best_adj = population[min_idx]
                
            # If we found a valid graph (0 conflicts of size 6), verify full
            if best_fitness == 0:
                if verify_full(n, best_adj):
                    return best_adj
            
            # Selection: Tournament Selection
            next_population = []
            for _ in range(pop_size):
                # Pick 3 random candidates, keep best
                idx1 = random.randint(0, pop_size-1)
                idx2 = random.randint(0, pop_size-1)
                idx3 = random.randint(0, pop_size-1)
                
                best_idx = idx1
                if fitnesses[idx2] < fitnesses[best_idx]:
                    best_idx = idx2
                if fitnesses[idx3] < fitnesses[best_idx]:
                    best_idx = idx3
                    
                # Crossover: Uniform crossover
                p1 = population[best_idx]
                p2 = population[random.randint(0, pop_size-1)]
                
                child = [0] * n
                for i in range(n):
                    for j in range(i+1, n):
                        # Randomly pick edge from p1 or p2
                        if random.random() < 0.5:
                            if p1[i] & (1 << j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                        else:
                            if p2[i] & (1 << j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                                
                # Mutation: Flip random edges
                # Probability of mutation per edge
                # For n=21, ~210 edges.
                # Let's flip ~5% of edges
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.05:
                            if child[i] & (1 << j):
                                child[i] ^= (1 << j)
                                child[j] ^= (1 << i)
                            else:
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                                
                next_population.append(child)
                
            population = next_population
            
        return best_adj

    # Try n=21
    # The GA might be too slow for n=21 with Python loops.
    # Let's try n=20 first as a fallback, but aim for 21.
    # Actually, let's just run GA for n=21. If it times out, we return the best found.
    # But we must return a VALID graph.
    # If GA doesn't find a valid one, we should fall back to a known valid construction or a faster local search.
    
    # Let's combine: Run GA for a short time. If it finds a 0-conflict graph, verify.
    # If not, run Local Search (Repair) on the best GA individual.
    
    best_adj_21 = ga_search(21, pop_size=30, max_generations=10)
    
    if best_adj_21 is not None:
        # Check if it's valid
        if verify_full(21, best_adj_21):
            return {'n': 21, 'edges': adj_to_edges(21, best_adj_21)}
        
        # If not valid, try to repair it using local search
        # Local Search: Flip edges to reduce conflicts
        adj = best_adj_21
        for _ in range(500): # 500 steps
            if time.time() > deadline - 10: # Leave 10s for fallback
                break
            # Find a conflict
            # We only checked size 6 in GA. Let's find a size 6 conflict.
            # If no size 6 conflict, check larger.
            # For repair, it's easier to just pick a random edge and flip if it reduces conflicts?
            # Or find a specific conflict.
            
            # Let's find a size 6 conflict
            conflict_found = False
            for S in combinations(range(21), 6):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                degs = [(adj[v] & mask).bit_count() for v in S]
                if len(set(degs)) == 1:
                    # Found conflict S
                    # Flip an edge inside S
                    u, v = random.sample(S, 2)
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    conflict_found = True
                    break
            
            if not conflict_found:
                # No size 6 conflicts. Check if full valid.
                if verify_full(21, adj):
                    return {'n': 21, 'edges': adj_to_edges(21, adj)}
                # If not full valid, flip a random edge globally
                i, j = random.sample(range(21), 2)
                if adj[i] & (1 << j):
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
    # Fallback: Try n=20 with the same GA/Repair strategy
    best_adj_20 = ga_search(20, pop_size=30, max_generations=10)
    
    if best_adj_20 is not None:
        if verify_full(20, best_adj_20):
            return {'n': 20, 'edges': adj_to_edges(20, best_adj_20)}
        
        adj = best_adj_20
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            conflict_found = False
            for S in combinations(range(20), 6):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                degs = [(adj[v] & mask).bit_count() for v in S]
                if len(set(degs)) == 1:
                    u, v = random.sample(S, 2)
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    conflict_found = True
                    break
            
            if not conflict_found:
                if verify_full(20, adj):
                    return {'n': 20, 'edges': adj_to_edges(20, adj)}
                i, j = random.sample(range(20), 2)
                if adj[i] & (1 << j):
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
    # Last resort: Trivial valid graph (n=5)
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
