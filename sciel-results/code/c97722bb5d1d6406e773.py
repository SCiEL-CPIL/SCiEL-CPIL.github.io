import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_limit):
        edges = generate_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        # Multi-start greedy
        # We will try different strategies for ordering neighbors
        # Strategy 1: Deterministic (by degree)
        # Strategy 2: Randomized shuffles for ties or full shuffle
        
        def run_greedy(shuffle_neighbors=False, seed_val=0):
            if shuffle_neighbors:
                rng_local = random.Random(seed_val)
            else:
                rng_local = None
                
            # Copy state
            rem_edges = set(edges)
            adj = [set() for _ in range(n)]
            for u, v in rem_edges:
                adj[u].add(v)
                adj[v].add(u)
            deg = [len(adj[i]) for i in range(n)]
            
            cliques = []
            
            while rem_edges:
                # Find vertex with max degree
                max_deg = -1
                v = -1
                for i in range(n):
                    if deg[i] > max_deg:
                        max_deg = deg[i]
                        v = i
                
                if v == -1 or max_deg == 0:
                    break
                
                # Get neighbors of v
                neighbors = list(adj[v])
                
                # Sort neighbors by degree in remaining graph
                # If shuffle_neighbors, add small random noise or shuffle ties
                if shuffle_neighbors:
                    # Shuffle neighbors with same degree? Or just shuffle all?
                    # Shuffling all might be too destructive.
                    # Let's shuffle neighbors with equal degrees.
                    # Simpler: Sort by (-deg, random_key)
                    keys = [(-deg[u], rng_local.random()) for u in neighbors]
                    neighbors = [u for _, u in sorted(zip(keys, neighbors))]
                else:
                    neighbors.sort(key=lambda x: -deg[x])
                
                # Greedy clique expansion
                clique = [v]
                for u in neighbors:
                    # Check if u is connected to all in clique
                    # clique[0] is v, so u is connected to v by definition (it's a neighbor)
                    # Check against clique[1:]
                    valid = True
                    for w in clique[1:]:
                        if (min(u, w), max(u, w)) not in rem_edges:
                            valid = False
                            break
                    if valid:
                        clique.append(u)
                
                # Add clique to result
                cliques.append(clique)
                
                # Remove edges in clique
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, w = clique[i], clique[j]
                        if u > w:
                            u, w = w, u
                        rem_edges.discard((u, w))
                        adj[u].discard(w)
                        adj[w].discard(u)
                        deg[u] -= 1
                        deg[w] -= 1
                        
            return cliques

        # Run deterministic first
        cliques = run_greedy(shuffle_neighbors=False)
        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques
            
        # Run randomized versions if time permits
        attempts = 0
        while time.time() < deadline - 5: # Leave 5s buffer
            attempts += 1
            cliques = run_greedy(shuffle_neighbors=True, seed_val=attempts)
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
            # Optional: break if we hit a very good solution?
            # For n=32, edges ~ 300. Cliques ~ 30-50.
            # For n=64, edges ~ 1000. Cliques ~ 100-200.
            # If we get significantly better, keep going.
            
        return best_cliques

    g1 = solve_instance(32, 0.6, 601, deadline)
    if time.time() > deadline - 5:
         # If we ran out of time, return what we have for g1 and empty for g2? 
         # No, we must return valid partitions.
         # If we didn't finish g2, we need a fast fallback for g2.
         # But g1 should be fast. Let's structure to ensure both are done.
         pass

    g2 = solve_instance(64, 0.55, 602, deadline)
    
    return [g1, g2]
