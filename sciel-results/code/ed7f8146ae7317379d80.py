"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 40
    
    # Helper to check if adding edge (u, v) creates a C6
    def creates_c6(adj, u, v):
        # Path u - x1 - x2 - x3 - x4 - v
        # x1 in N(u), x4 in N(v)
        # If x1 == x4, need triangle x1-x2-x3-x1
        # If x1 != x4, need path x1-x2-x3-x4
        
        neighbors_u = [x for x in range(n) if adj[u][x] and x != v]
        neighbors_v = [x for x in range(n) if adj[v][x] and x != u]
        
        for x1 in neighbors_u:
            for x4 in neighbors_v:
                if x1 == x4:
                    # Check for triangle containing x1
                    # x1 - x2 - x3 - x1
                    # x2 in N(x1), x3 in N(x2) cap N(x1)
                    # And x2, x3 distinct from u, v, x1
                    for x2 in range(n):
                        if x2 in (u, v, x1):
                            continue
                        if not adj[x1][x2]:
                            continue
                        for x3 in range(n):
                            if x3 in (u, v, x1, x2):
                                continue
                            if adj[x2][x3] and adj[x3][x1]:
                                return True
                else:
                    # Check for path x1 - x2 - x3 - x4
                    for x2 in range(n):
                        if x2 in (u, v, x1, x4):
                            continue
                        if not adj[x1][x2]:
                            continue
                        for x3 in range(n):
                            if x3 in (u, v, x1, x2, x4):
                                continue
                            if adj[x2][x3] and adj[x3][x4]:
                                return True
        return False

    def remove_c6s(adj, edges_set):
        # Iterate through edges and remove if in C6
        # To be efficient, we check if edge (u,v) is in a C6
        # This means there is a path of length 4 between u and v in G \ {(u,v)}
        # This is slightly different from creates_c6 which checks if ADDING (u,v) creates C6.
        # If (u,v) is in G, and we check creates_c6(u,v) on G \ {(u,v)}, it tells us if (u,v) completes a C6.
        # So we can just use creates_c6 on the graph WITHOUT the edge.
        
        to_remove = []
        for u, v in list(edges_set):
            # Temporarily remove
            adj[u][v] = False
            adj[v][u] = False
            
            if creates_c6(adj, u, v):
                to_remove.append((u, v))
            
            # Restore
            adj[u][v] = True
            adj[v][u] = True
        
        for u, v in to_remove:
            adj[u][v] = False
            adj[v][u] = False
            edges_set.discard((u, v))
            edges_set.discard((v, u))

    best_edges = []
    best_count = 0
    
    # Try multiple random starts
    for seed in range(100):
        if time.time() > deadline - 20.0:
            break
            
        random.seed(seed)
        adj = [[False] * n for _ in range(n)]
        edges_set = set()
        
        # Initialize with random graph p=0.5
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u][v] = True
                    adj[v][u] = True
                    edges_set.add((u, v))
                    edges_set.add((v, u))
        
        # Remove C6s
        remove_c6s(adj, edges_set)
        
        # Greedy add
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u][v]:
                        if not creates_c6(adj, u, v):
                            adj[u][v] = True
                            adj[v][u] = True
                            edges_set.add((u, v))
                            edges_set.add((v, u))
                            changed = True
                            
        current_count = len(edges_set) // 2
        if current_count > best_count:
            best_count = current_count
            best_edges = [[u, v] for u, v in edges_set if u < v]
            
        # Local Search: 2-Swap
        # Try to remove 2 edges and add 2 edges
        # We look for a pair of missing edges (a,b), (c,d) and present edges (e,f), (g,h)
        # Such that removing (e,f),(g,h) and adding (a,b),(c,d) is C6-free
        
        # Simplified: Try to add one edge, if fails, try to remove one edge and add two.
        # This is complex. Let's just do 1-swap (add edge) which we already did.
        # Let's try a specific 2-swap heuristic:
        # Pick a random missing edge (u,v). If it creates C6, find the C6 and try to break it.
        
        if time.time() > deadline - 10.0:
            break
            
        # Perform local search for 10 seconds
        ls_start = time.time()
        improved = True
        while improved and time.time() < ls_start + 10.0:
            improved = False
            # Try all pairs of edges to remove
            # This is O(E^2 * Check) which is too slow.
            # Instead, try random swaps.
            
            # Pick a random edge to remove
            if len(edges_set) == 0: break
            e1 = random.choice(list(edges_set))
            if e1[0] > e1[1]: e1 = (e1[1], e1[0])
            
            # Pick a random missing edge to add
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u][v]]
            if not missing: continue
            e2 = random.choice(missing)
            
            # Remove e1
            u1, v1 = e1
            adj[u1][v1] = False
            adj[v1][u1] = False
            edges_set.discard(e1)
            edges_set.discard((v1, u1))
            
            # Try to add e2
            u2, v2 = e2
            if not creates_c6(adj, u2, v2):
                adj[u2][v2] = True
                adj[v2][u2] = True
                edges_set.add(e2)
                edges_set.add((v2, u2))
                improved = True
            else:
                # Restore e1
                adj[u1][v1] = True
                adj[v1][u1] = True
                edges_set.add(e1)
                edges_set.add((v1, u1))
                
        current_count = len(edges_set) // 2
        if current_count > best_count:
            best_count = current_count
            best_edges = [[u, v] for u, v in edges_set if u < v]

    return best_edges
# EVOLVE-BLOCK-END
