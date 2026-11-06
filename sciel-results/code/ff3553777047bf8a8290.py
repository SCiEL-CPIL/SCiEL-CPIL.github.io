"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Construct a graph with 31 vertices and 91 edges minimizing the number of 4-cycles.
    
    Strategy: Greedy Construction + Systematic Local Search.
    1. Greedy Build: Add edges one by one, always choosing the pair with the fewest common neighbors.
    2. Local Search: Systematically scan all edges. For each edge, find the best non-edge to swap with.
       Accept the swap if it reduces the total number of 4-cycles.
    3. Use a time limit to ensure we finish within 120s.
    """
    n = 31
    target_edges = 91
    
    # Initialize empty graph
    adj = [set() for _ in range(n)]
    edge_set = set()
    
    # Helper to get common neighbors count
    def common_neighbors(u, v):
        return len(adj[u] & adj[v])
    
    # 1. Greedy Initialization
    for _ in range(target_edges):
        best_pair = None
        min_c4s = float('inf')
        # Iterate all non-edges
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in edge_set:
                    continue
                c4s = common_neighbors(u, v)
                if c4s < min_c4s:
                    min_c4s = c4s
                    best_pair = (u, v)
        
        if best_pair:
            u, v = best_pair
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            
    # 2. Local Search
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    while time.time() - start_time < time_limit:
        improved = False
        # Get list of edges to iterate. 
        # We iterate over a snapshot to avoid modification issues during iteration,
        # but if we accept a swap, we break and restart the sweep.
        current_edges = list(edge_set)
        
        for u, v in current_edges:
            # Check if (u, v) is still in edge_set
            if (u, v) not in edge_set:
                continue
                
            rem_c4s = common_neighbors(u, v)
            
            # Temporarily remove (u, v)
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.remove((u, v))
            
            best_add_pair = None
            min_add_c4s = float('inf')
            
            # Find best non-edge to add
            for x in range(n):
                for y in range(x + 1, n):
                    if (x, y) in edge_set:
                        continue
                    add_c4s = common_neighbors(x, y)
                    if add_c4s < min_add_c4s:
                        min_add_c4s = add_c4s
                        best_add_pair = (x, y)
            
            if best_add_pair and min_add_c4s < rem_c4s:
                # Accept swap
                x, y = best_add_pair
                adj[x].add(y)
                adj[y].add(x)
                edge_set.add((x, y))
                improved = True
                break # Restart sweep
            else:
                # Revert removal
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
        
        if not improved:
            # No improvement in full sweep. 
            # Try a random perturbation to escape local minimum.
            stuck = True
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                # Pick random edge to remove
                u, v = random.choice(list(edge_set))
                rem_c4s = common_neighbors(u, v)
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.remove((u, v))
                
                # Pick random non-edge to add
                while True:
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x == y: continue
                    if x > y: x, y = y, x
                    if (x, y) not in edge_set:
                        break
                
                add_c4s = common_neighbors(x, y)
                
                # Accept if better
                if add_c4s < rem_c4s:
                    adj[x].add(y)
                    adj[y].add(x)
                    edge_set.add((x, y))
                    stuck = False
                    break
                else:
                    # Revert
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
            
            if stuck:
                # If we couldn't improve even with random perturbations, break.
                break

    # Convert to list of lists
    result = [[u, v] for u, v in edge_set]
    
    return result

# EVOLVE-BLOCK-END
