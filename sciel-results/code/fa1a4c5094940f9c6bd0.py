import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def validate_and_add(adj, u, v):
        """Check if edge (u,v) can be added given current adj. Returns True if valid."""
        # Condition 1: u and v must have at most 2 common neighbors
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # Condition 2: For each neighbor b of v (in current graph), 
        # u and b must have at most 1 common neighbor
        mask = adj[v]
        while mask:
            b = (mask & -mask).bit_length() - 1
            if (adj[u] & adj[b]).bit_count() > 1:
                return False
            mask &= mask - 1
        
        # Condition 3: For each neighbor a of u (in current graph),
        # v and a must have at most 1 common neighbor
        mask = adj[u]
        while mask:
            a = (mask & -mask).bit_length() - 1
            if (adj[v] & adj[a]).bit_count() > 1:
                return False
            mask &= mask - 1
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    def local_search(adj, max_iter=100):
        """Improve adj by adding edges and swapping."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
            
            # Try adding any missing edge
            # We can just iterate all possible edges
            for u, v in all_edges:
                if not (adj[u] & (1 << v)): # Not present
                    if validate_and_add(adj, u, v):
                        add_edge(adj, u, v)
                        improved = True
                        # Don't break, try to add more in this pass? 
                        # Breaking might be faster if we want to restart scan, 
                        # but adding all possible is better.
                        # However, adding one might block others.
                        # Let's just add one and break to restart scan for safety/simplicity.
                        break
            
            # If no edge added, try swaps
            if not improved:
                # Try to remove one edge and add two
                # Randomly pick an edge to remove
                current_edges = [(u, v) for u, v in all_edges if (adj[u] & (1 << v))]
                if current_edges:
                    u, v = random.choice(current_edges)
                    old_adj = adj[:]
                    remove_edge(adj, u, v)
                    
                    added = 0
                    # Try to add edges incident to u or v, or others
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if not (adj[u] & (1 << w)) and validate_and_add(adj, u, w):
                            add_edge(adj, u, w)
                            added += 1
                        if not (adj[v] & (1 << w)) and validate_and_add(adj, v, w):
                            add_edge(adj, v, w)
                            added += 1
                        if added >= 2:
                            break
                    
                    if added < 2:
                        # Revert
                        adj[:] = old_adj
                    else:
                        improved = True
        
        return adj

    def greedy_construct():
        """Create a random greedy solution."""
        adj = [0] * n
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)
        for u, v in edges_copy:
            if validate_and_add(adj, u, v):
                add_edge(adj, u, v)
        return adj

    def crossover(adj1, adj2):
        """Merge adj1 and adj2 by adding edges from adj2 to adj1 if valid."""
        child = adj1[:]
        for u, v in all_edges:
            if (adj2[u] & (1 << v)) and not (child[u] & (1 << v)):
                if validate_and_add(child, u, v):
                    add_edge(child, u, v)
        return child

    def mutate(adj):
        """Randomly remove an edge and try to add a new one."""
        child = adj[:]
        current_edges = [(u, v) for u, v in all_edges if (child[u] & (1 << v))]
        if current_edges:
            u, v = random.choice(current_edges)
            remove_edge(child, u, v)
            
        # Try to add a random missing edge
        missing = [(u, v) for u, v in all_edges if not (child[u] & (1 << v))]
        if missing:
            u, v = random.choice(missing)
            if validate_and_add(child, u, v):
                add_edge(child, u, v)
                
        return child

    # Main GA Loop
    pop_size = 50
    population = []
    
    # Initialize population
    for _ in range(pop_size):
        adj = greedy_construct()
        adj = local_search(adj, max_iter=20)
        population.append(adj)
    
    best_adj = max(population, key=count_edges)
    best_count = count_edges(best_adj)
    
    gen = 0
    while time.time() < deadline:
        gen += 1
        new_population = []
        
        # Elitism: keep top 10%
        population.sort(key=count_edges, reverse=True)
        elite_count = max(1, pop_size // 10)
        new_population.extend(population[:elite_count])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
                
            # Tournament selection
            p1 = random.choice(population[:pop_size//2])
            p2 = random.choice(population[:pop_size//2])
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            if random.random() < 0.5:
                child = mutate(child)
            
            # Local Search
            child = local_search(child, max_iter=20)
            
            new_population.append(child)
            
            # Update best
            c = count_edges(child)
            if c > best_count:
                best_count = c
                best_adj = child[:]
        
        population = new_population
        
        # Periodically re-evaluate best with deeper local search?
        if gen % 10 == 0:
             best_adj = local_search(best_adj, max_iter=100)
             best_count = count_edges(best_adj)

    # Convert best_adj to list of edges
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                best_edges.append([u, v])
                
    return best_edges
