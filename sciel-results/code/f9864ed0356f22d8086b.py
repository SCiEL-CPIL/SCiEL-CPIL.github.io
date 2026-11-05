import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    N = 20
    # adj[i] is a set of neighbors
    adj = [set() for _ in range(N)]
    # common[i][j] is the number of common neighbors
    common = [[0] * N for _ in range(N)]
    
    def add_edge(u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        
        # Check validity
        # For each w in adj[v], common[u][w] must be < 2
        # For each w in adj[u], common[v][w] must be < 2
        
        # Optimization: Check before updating
        for w in adj[v]:
            if common[u][w] >= 2:
                return False
        for w in adj[u]:
            if common[v][w] >= 2:
                return False
                
        # Update adjacency
        adj[u].add(v)
        adj[v].add(u)
        
        # Update common counts
        # For each w in adj[v] (before adding u to adj[v]? No, adj[v] already has its neighbors. 
        # When we add u-v, u becomes a neighbor of v.
        # The common neighbors of u and w (where w in adj[v]) increase by 1 because v is now a common neighbor?
        # Wait.
        # Common neighbors of u and w are vertices x such that x is neighbor of u and x is neighbor of w.
        # Before adding u-v:
        #   N(u) did not contain v.
        #   N(w) contained v (since w in adj[v] means v in adj[w]).
        #   So v was NOT a common neighbor of u and w (since v not in N(u)).
        # After adding u-v:
        #   N(u) contains v.
        #   N(w) contains v.
        #   So v IS a common neighbor of u and w.
        #   Thus common[u][w] increases by 1.
        
        for w in adj[v]:
            common[u][w] += 1
            common[w][u] += 1
            
        for w in adj[u]:
            common[v][w] += 1
            common[w][v] += 1
            
        return True

    def remove_edge(u, v):
        if v not in adj[u]:
            return False
            
        adj[u].remove(v)
        adj[v].remove(u)
        
        for w in adj[v]:
            common[u][w] -= 1
            common[w][u] -= 1
            
        for w in adj[u]:
            common[v][w] -= 1
            common[w][v] -= 1
            
        return True

    def get_edges():
        edges = []
        for u in range(N):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    def count_edges():
        return sum(len(adj[u]) for u in range(N)) // 2

    best_edges = []
    best_count = 0
    
    # Helper to reset state
    def reset_state(edges_list):
        nonlocal adj, common
        adj = [set() for _ in range(N)]
        common = [[0] * N for _ in range(N)]
        for u, v in edges_list:
            add_edge(u, v)

    # Strategy 1: Start with 5 K4s
    base_edges = []
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                base_edges.append([4 * k + a, 4 * k + b])
                
    reset_state(base_edges)
    best_edges = get_edges()
    best_count = count_edges()
    
    # Strategy 2: Randomized Greedy from scratch or from base
    # We will try multiple restarts.
    # In each restart, we try to improve the current best or a random start.
    
    max_restarts = 5000 # Will be limited by time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Decide start state: sometimes start from best, sometimes from random/empty
        if restart % 5 == 0:
            # Start from best
            reset_state(best_edges)
        else:
            # Start from random sparse graph
            adj = [set() for _ in range(N)]
            common = [[0] * N for _ in range(N)]
            # Add random edges
            possible_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
            random.shuffle(possible_edges)
            for u, v in possible_edges[:20]: # Start with ~20 edges
                if add_edge(u, v):
                    pass
        
        # Greedy Fill Phase
        # Try to add any valid missing edge
        # We can iterate through all pairs and try to add
        # To be faster, we can just try all pairs in a random order
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Generate list of missing edges
            missing = [(u, v) for u in range(N) for v in range(u + 1, N) if v not in adj[u]]
            random.shuffle(missing)
            for u, v in missing:
                if add_edge(u, v):
                    improved = True
                    break # Restart the while loop to re-shuffle? Or just continue?
                    # Continuing is better to catch multiple additions in one pass
                    # But re-shuffling helps avoid bias.
                    # Let's just continue in the loop for speed, and do a full pass if no improvement
                    # Actually, if we add an edge, the set of missing edges changes.
                    # So we should break and re-generate missing edges.
        
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_edges = get_edges()
            
        # Local Search / Perturbation
        # Try to remove a random edge and see if we can add more
        if time.time() > deadline:
            break
            
        # Pick a random existing edge to remove
        existing = [(u, v) for u in range(N) for v in adj[u] if u < v]
        if existing:
            u, v = random.choice(existing)
            remove_edge(u, v)
            
            # Try to add edges again
            improved = True
            while improved and time.time() < deadline:
                improved = False
                missing = [(x, y) for x in range(N) for y in range(x + 1, N) if y not in adj[x]]
                random.shuffle(missing)
                for x, y in missing:
                    if add_edge(x, y):
                        improved = True
                        break
                        
            current_count = count_edges()
            if current_count > best_count:
                best_count = current_count
                best_edges = get_edges()
            else:
                # Revert? 
                # We modified adj/common. 
                # We need to restore the state before the removal.
                # It's easier to just keep the state if it's not worse, but if it's worse, 
                # we should probably revert to the previous best state for the next iteration?
                # Or just continue from this state?
                # Hill climbing usually keeps the state if it's not worse, or reverts.
                # Since we are doing restarts, we can just let it be.
                # But if we don't revert, we might drift away.
                # Let's revert to best_edges if this state is worse?
                # No, that kills diversity.
                # Let's just keep the current state for the next iteration of the outer loop?
                # The outer loop restarts from best_edges every 5th time.
                # So it's fine.
                pass

    return best_edges
