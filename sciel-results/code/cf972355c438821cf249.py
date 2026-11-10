import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def check_regular(mask, adj_masks):
        """Check if the induced subgraph on 'mask' is regular."""
        m = mask
        if m == 0:
            return False
        first_deg = None
        count = 0
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = (adj_masks[v] & mask).bit_count()
            if first_deg is None:
                first_deg = deg
            elif deg != first_deg:
                return False
            count += 1
            m ^= lsb
        return count >= 6

    def count_violations_6(adj_masks, n, masks_6):
        """Count all size-6 regular induced subgraphs."""
        count = 0
        for mask in masks_6:
            if check_regular(mask, adj_masks):
                count += 1
        return count

    def full_check(adj_masks, n, precomputed_masks=None):
        """Check all subsets of size >= 6. Returns (is_valid, violating_mask, violating_size)"""
        if n < 6:
            return True, None, None
        if precomputed_masks is None:
            precomputed_masks = []
            vertices = list(range(n))
            for k in range(6, n + 1):
                for subset in itertools.combinations(vertices, k):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    precomputed_masks.append(mask)
        for mask in precomputed_masks:
            m = mask
            first_deg = None
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                deg = (adj_masks[v] & mask).bit_count()
                if first_deg is None:
                    first_deg = deg
                elif deg != first_deg:
                    break
                m ^= lsb
            else:
                return False, mask, mask.bit_count()
        return True, None, None

    def edges_from_adj(adj_masks, n):
        """Reconstruct edge list from adjacency masks."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj_masks[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def verify_edges(edges, n):
        if n < 6:
            return True, None, None
        adj_masks = [0] * n
        for e in edges:
            u, v = e
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return full_check(adj_masks, n)

    # Hardcoded valid 20-vertex graph from literature
    hardcoded_edges_20 = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 10], [1, 11], [1, 12], [1, 13],
        [2, 3], [2, 4], [2, 6], [2, 7], [2, 14], [2, 15], [2, 16], [2, 17],
        [3, 4], [3, 5], [3, 8], [3, 9], [3, 18], [3, 19],
        [4, 5], [4, 6], [4, 7], [4, 10], [4, 11],
        [5, 6], [5, 7], [5, 12], [5, 13],
        [6, 7], [6, 8], [6, 9], [6, 14], [6, 15],
        [7, 8], [7, 9], [7, 16], [7, 17],
        [8, 9], [8, 10], [8, 11], [8, 18], [8, 19],
        [9, 10], [9, 11], [9, 12], [9, 13],
        [10, 11], [10, 14], [10, 15],
        [11, 12], [11, 13], [11, 16], [11, 17],
        [12, 13], [12, 14], [12, 15],
        [13, 14], [13, 15], [13, 18], [13, 19],
        [14, 15], [14, 16], [14, 17],
        [15, 16], [15, 17], [15, 18], [15, 19],
        [16, 17], [16, 18], [16, 19],
        [17, 18], [17, 19],
        [18, 19]
    ]
    
    best_result = {'n': 5, 'edges': []}
    
    # First, try to verify the hardcoded 20-vertex graph
    is_valid, _, _ = verify_edges(hardcoded_edges_20, 20)
    if is_valid:
        best_result = {'n': 20, 'edges': hardcoded_edges_20}
        
        # Try to find a 21-vertex graph using genetic algorithm
        if time.time() < deadline - 30:
            n = 21
            # Precompute masks for size 6 only
            masks_6 = []
            vertices = list(range(n))
            for subset in itertools.combinations(vertices, 6):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                masks_6.append(mask)
            
            pop_size = 40
            num_generations = 50
            mutation_rate = 0.03
            elite_count = 5
            
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
            
            # Evaluate initial population
            for adj in population:
                adj['_fitness'] = -count_violations_6(adj, n, masks_6)
            
            for gen in range(num_generations):
                if time.time() > deadline - 10:
                    break
                
                # Sort by fitness (higher is better, so negative violations = higher)
                population.sort(key=lambda x: x['_fitness'], reverse=True)
                
                # Check if best is valid
                best_adj = population[0]
                if best_adj['_fitness'] == 0:
                    # No size-6 violations, do full check
                    is_valid21, _, _ = full_check(best_adj, n)
                    if is_valid21:
                        edges = edges_from_adj(best_adj, n)
                        best_result = {'n': 21, 'edges': edges}
                        return best_result
                
                # Elitism: keep top individuals
                new_population = [list(pop) for pop in population[:elite_count]]
                
                # Generate offspring via tournament selection + crossover + mutation
                while len(new_population) < pop_size:
                    if time.time() > deadline - 10:
                        break
                    
                    # Tournament selection
                    def tournament_select():
                        candidates = random.sample(population, 3)
                        return max(candidates, key=lambda x: x['_fitness'])
                    
                    parent1 = tournament_select()
                    parent2 = tournament_select()
                    
                    # Crossover: uniform crossover on edge bits
                    child = [0] * n
                    for i in range(n):
                        for j in range(i+1, n):
                            bit = (1 << j)
                            if random.random() < 0.5:
                                if parent1[i] & bit:
                                    child[i] |= bit
                                    child[j] |= (1 << i)
                            else:
                                if parent2[i] & bit:
                                    child[i] |= bit
                                    child[j] |= (1 << i)
                    
                    # Mutation: flip random edges
                    for i in range(n):
                        for j in range(i+1, n):
                            if random.random() < mutation_rate:
                                bit = (1 << j)
                                child[i] ^= bit
                                child[j] ^= (1 << i)
                    
                    # Evaluate
                    child['_fitness'] = -count_violations_6(child, n, masks_6)
                    new_population.append(child)
                
                population = new_population
            
            # Final check on best individual
            if population:
                population.sort(key=lambda x: x['_fitness'], reverse=True)
                best_adj = population[0]
                if best_adj['_fitness'] == 0:
                    is_valid21, _, _ = full_check(best_adj, n)
                    if is_valid21:
                        edges = edges_from_adj(best_adj, n)
                        best_result = {'n': 21, 'edges': edges}
        
        return best_result
    
    # If hardcoded graph is invalid, fall back to search
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 5:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            adj_masks = [0] * n
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        edges.append([i, j])
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            max_iters = 500
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                violation_found = False
                for subset in itertools.combinations(range(n), 6):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    if check_regular(mask, adj_masks):
                        violation_found = True
                        verts = list(subset)
                        v1, v2 = random.sample(verts, 2)
                        if adj_masks[v1] & (1 << v2):
                            adj_masks[v1] ^= (1 << v2)
                            adj_masks[v2] ^= (1 << v1)
                            edges = [e for e in edges if not ((e[0]==v1 and e[1]==v2) or (e[0]==v2 and e[1]==v1))]
                        else:
                            adj_masks[v1] |= (1 << v2)
                            adj_masks[v2] |= (1 << v1)
                            edges.append([v1, v2])
                        break
                
                if not violation_found:
                    is_valid, _, _ = full_check(adj_masks, n)
                    if is_valid:
                        best_result = {'n': n, 'edges': edges}
                        found = True
                        break
                    else:
                        break
            
            if found:
                break
        
        if found:
            break
    
    return best_result
