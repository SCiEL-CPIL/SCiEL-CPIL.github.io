import random
import itertools
import time
import copy

def construct():
    def is_regular_subset(adj, n, subset):
        """Check if induced subgraph on subset is regular."""
        if len(subset) < 6:
            return False
        mask = 0
        for v in subset:
            mask |= (1 << v)
        degs = []
        for v in subset:
            deg = bin(adj[v] & mask).count('1')
            degs.append(deg)
        return len(set(degs)) == 1

    def count_violations_fast(adj, n, max_check=500):
        """Count violations by sampling. Returns (count, found_violation)."""
        violations = 0
        found = None
        for _ in range(max_check):
            k = random.randint(6, n)
            subset = random.sample(range(n), k)
            if is_regular_subset(adj, n, subset):
                violations += 1
                if found is None:
                    found = subset
        return violations, found

    def full_check(adj, n):
        """Exhaustive check."""
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                if is_regular_subset(adj, n, comb):
                    return False
        return True

    def evolve(n, time_limit=105):
        """Population-based evolutionary search."""
        start = time.time()
        pop_size = 30
        population = []
        
        # Initialize population with diverse graphs
        for i in range(pop_size):
            p = 0.1 + 0.8 * i / pop_size
            adj = [0] * n
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            population.append(adj)
        
        best_adj = None
        best_violations = float('inf')
        
        generation = 0
        while time.time() - start < time_limit:
            generation += 1
            
            # Evaluate population
            scored = []
            for adj in population:
                viol, _ = count_violations_fast(adj, n, 100)
                scored.append((viol, adj))
            
            scored.sort(key=lambda x: x[0])
            
            if scored[0][0] == 0:
                # Verify with full check
                if full_check(scored[0][1], n):
                    return scored[0][1]
            
            if scored[0][0] < best_violations:
                best_violations = scored[0][0]
                best_adj = scored[0][1]
            
            # Create next generation
            next_pop = []
            
            # Elitism: keep top 5
            for i in range(min(5, len(scored))):
                next_pop.append(copy.deepcopy(scored[i][1]))
            
            # Crossover and mutation
            while len(next_pop) < pop_size:
                # Select two parents (tournament selection)
                p1_idx = random.randint(0, min(10, len(scored)-1))
                p2_idx = random.randint(0, min(10, len(scored)-1))
                p1 = scored[p1_idx][1]
                p2 = scored[p2_idx][1]
                
                # Uniform crossover
                child = [0] * n
                for u in range(n):
                    for v in range(u+1, n):
                        if random.random() < 0.5:
                            child[u] |= (p1[u] & (1 << v))
                            child[v] |= (p1[v] & (1 << u))
                        else:
                            child[u] |= (p2[u] & (1 << v))
                            child[v] |= (p2[v] & (1 << u))
                
                # Mutation: flip random edges
                for _ in range(3):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        if (child[u] & (1 << v)) != 0:
                            child[u] &= ~(1 << v)
                            child[v] &= ~(1 << u)
                        else:
                            child[u] |= (1 << v)
                            child[v] |= (1 << u)
                
                next_pop.append(child)
            
            population = next_pop
        
        return best_adj

    # Try n=22, 21, 20
    for n in [22, 21, 20]:
        adj = evolve(n, time_limit=105 if n <= 21 else 50)
        if adj is not None:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] & (1 << j)) != 0:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    return {'n': 5, 'edges': []}
