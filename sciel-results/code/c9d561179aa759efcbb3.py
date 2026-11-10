<<<<<<< SEARCH
import time
import random
from itertools import combinations
from math import comb

START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # safety margin before 150s kill

def is_regular_subgraph(subset, adj_matrix, n):
    """Check if the induced subgraph on `subset` is regular.
    A regular subgraph means every vertex in the subset has the same degree
    within the subset (counting only edges to other vertices in the subset).
    """
    k = len(subset)
    if k < 2:
        # Single vertex: degree 0, trivially regular
        return True
    
    # Compute degrees of each vertex in the subset
    degrees = []
    for u in subset:
        deg = 0
        for v in subset:
            if u != v and adj_matrix[u][v]:
                deg += 1
        degrees.append(deg)
    
    # Check if all degrees are equal
    first_deg = degrees[0]
    for d in degrees[1:]:
        if d != first_deg:
            return False
    return True

def has_regular_subgraph_of_size_at_least_6(n, adj_matrix):
    """Check if there exists any subset of size >= 6 that induces a regular subgraph."""
    # We only need to check subsets of exactly size 6 (or more).
    # But checking all subsets of size >= 6 is expensive.
    # Key insight: if a subset of size m >= 6 induces a regular subgraph,
    # we can check subsets of size 6 first. But actually, a regular subgraph
    # of size m > 6 might exist without any size-6 regular subgraph within it.
    # However, the problem says "No set of 6 or more vertices may induce a regular subgraph."
    # So we need to check ALL subsets of size >= 6.
    # For efficiency, we'll check sizes 6, 7, ..., n.
    
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if is_regular_subgraph(subset, adj_matrix, n):
                return True
    return False

def validate_graph(n, edges):
    """Validate that the graph is simple and has no regular induced subgraph on >= 6 vertices."""
    # Build adjacency matrix
    adj_matrix = [[0] * n for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False  # no self-loops
        if not (0 <= u < n and 0 <= v < n):
            return False
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1
    
    # Check for duplicate edges
    edge_set = set()
    for u, v in edges:
        pair = (min(u, v), max(u, v))
        if pair in edge_set:
            return False  # duplicate edge
        edge_set.add(pair)
    
    # Check for regular subgraph of size >= 6
    if has_regular_subgraph_of_size_at_least_6(n, adj_matrix):
        return False
    
    return True

def construct():
    # We want to maximize n such that there exists a graph on n vertices
    # with no regular induced subgraph on 6 or more vertices.
    
    # Known: graphs with 20 vertices exist. Let's try to find one.
    # Strategy: Start with n=20 and try random graphs, or use a constructive approach.
    
    # A simple approach: try random graphs for n=20, 21, etc.
    # But validation is expensive for large n. Let's be smart.
    
    # For n=20, the number of subsets of size 6 is comb(20,6) = 38760, which is manageable.
    # For n=21, comb(21,6) = 54264, still manageable.
    # For n=22, comb(22,6) = 74613, still OK.
    
    # Let's try n=22 first, then 21, then 20.
    
    for n in [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > DEADLINE - 10:
            break
        
        # Try random graphs
        max_attempts = 50 if n <= 20 else 20
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 5:
                break
            
            # Generate a random graph
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        edges.append([u, v])
            
            # Validate
            if validate_graph(n, edges):
                return {'n': n, 'edges': edges}
        
        # If random didn't work, try a structured approach for smaller n
        # For n <= 15, we can do a more exhaustive search if needed
        if n <= 15:
            # Try complement of random graph
            for attempt in range(20):
                if time.time() > DEADLINE - 5:
                    break
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            edges.append([u, v])
                if validate_graph(n, edges):
                    return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
=======
import time
import random
from itertools import combinations
from math import comb

START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # safety margin before 150s kill

def count_regular_subgraphs_size_6(adj_matrix, n):
    """Count the number of 6-vertex subsets that induce a regular subgraph."""
    count = 0
    for subset in combinations(range(n), 6):
        degrees = []
        for u in subset:
            deg = 0
            for v in subset:
                if u != v and adj_matrix[u][v]:
                    deg += 1
            degrees.append(deg)
        if len(set(degrees)) == 1:
            count += 1
    return count

def validate_no_regular_6plus(n, adj_matrix):
    """Check if no subset of size >= 6 induces a regular subgraph."""
    # First check size 6 (most common violation)
    for subset in combinations(range(n), 6):
        degrees = []
        for u in subset:
            deg = 0
            for v in subset:
                if u != v and adj_matrix[u][v]:
                    deg += 1
            degrees.append(deg)
        if len(set(degrees)) == 1:
            return False
    
    # Check sizes 7 through n (less likely but must verify)
    for size in range(7, n + 1):
        for subset in combinations(range(n), size):
            degrees = []
            for u in subset:
                deg = 0
                for v in subset:
                    if u != v and adj_matrix[u][v]:
                        deg += 1
                degrees.append(deg)
            if len(set(degrees)) == 1:
                return False
    
    return True

def build_adj_matrix(n, edges):
    """Build adjacency matrix from edge list."""
    adj = [[0] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj

def edges_from_adj(adj, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append([u, v])
    return edges

def crossover(parent1_adj, parent2_adj, n):
    """Uniform crossover: each edge position is taken from one parent or the other."""
    child_adj = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.5:
                child_adj[u][v] = parent1_adj[u][v]
                child_adj[v][u] = parent1_adj[u][v]
            else:
                child_adj[u][v] = parent2_adj[u][v]
                child_adj[v][u] = parent2_adj[u][v]
    return child_adj

def mutate(adj, n, mutation_rate=0.05):
    """Flip random edges with given probability."""
    child_adj = [row[:] for row in adj]
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < mutation_rate:
                child_adj[u][v] = 1 - child_adj[u][v]
                child_adj[v][u] = child_adj[u][v]
    return child_adj

def tournament_selection(population, fitnesses, k=3):
    """Select the best individual from a random tournament of size k."""
    indices = random.sample(range(len(population)), k)
    best_idx = min(indices, key=lambda i: fitnesses[i])
    return population[best_idx]

def genetic_algorithm(n, pop_size=30, generations=50, time_limit=140.0):
    """Run genetic algorithm to find a graph on n vertices with no regular induced subgraph on 6+ vertices."""
    # Initialize population with random graphs
    population = []
    for _ in range(pop_size):
        adj = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u][v] = 1
                    adj[v][u] = 1
        population.append(adj)
    
    # Compute initial fitnesses (lower is better: 0 means valid)
    fitnesses = []
    for adj in population:
        fitness = count_regular_subgraphs_size_6(adj, n)
        fitnesses.append(fitness)
    
    best_fitness = min(fitnesses)
    best_adj = population[fitnesses.index(best_fitness)]
    
    # Check if any individual is already valid
    if best_fitness == 0:
        if validate_no_regular_6plus(n, best_adj):
            return edges_from_adj(best_adj, n)
    
    # Evolution loop
    for gen in range(generations):
        if time.time() > time_limit:
            break
        
        new_population = []
        
        # Elitism: keep the best individual
        best_idx = fitnesses.index(min(fitnesses))
        new_population.append([row[:] for row in population[best_idx]])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Select parents via tournament
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child = crossover(parent1, parent2, n)
            
            # Mutation
            child = mutate(child, n, mutation_rate=0.03)
            
            new_population.append(child)
        
        # Evaluate new population
        new_fitnesses = []
        for adj in new_population:
            fitness = count_regular_subgraphs_size_6(adj, n)
            new_fitnesses.append(fitness)
        
        # Check if any individual is valid
        min_fit = min(new_fitnesses)
        if min_fit == 0:
            valid_idx = new_fitnesses.index(0)
            if validate_no_regular_6plus(n, new_population[valid_idx]):
                return edges_from_adj(new_population[valid_idx], n)
        
        # Update population
        population = new_population
        fitnesses = new_fitnesses
        
        # Track best
        if min(fitnesses) < best_fitness:
            best_fitness = min(fitnesses)
            best_adj = population[fitnesses.index(best_fitness)]
        
        # Early termination if stuck
        if best_fitness > 0 and gen > 20 and all(f > 0 for f in fitnesses):
            # Try some random restarts
            for _ in range(5):
                adj = [[0] * n for _ in range(n)]
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            adj[u][v] = 1
                            adj[v][u] = 1
                fitness = count_regular_subgraphs_size_6(adj, n)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_adj = adj
                if fitness == 0:
                    if validate_no_regular_6plus(n, adj):
                        return edges_from_adj(adj, n)
    
    # Return best found even if not valid (caller will check)
    return edges_from_adj(best_adj, n)

def construct():
    """Find a graph with no regular induced subgraph on 6+ vertices, maximizing n."""
    # Try n from 22 down to 10
    for n in [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > DEADLINE - 15:
            break
        
        # Run genetic algorithm
        edges = genetic_algorithm(n, pop_size=25, generations=40, time_limit=DEADLINE - 10)
        
        if edges is not None:
            # Validate the result
            adj = build_adj_matrix(n, edges)
            if validate_no_regular_6plus(n, adj):
                return {'n': n, 'edges': edges}
        
        # If GA didn't find a valid graph, try a few random graphs as fallback
        for _ in range(10):
            if time.time() > DEADLINE - 5:
                break
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        edges.append([u, v])
            adj = build_adj_matrix(n, edges)
            if validate_no_regular_6plus(n, adj):
                return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
>>>>>>> REPLACE
