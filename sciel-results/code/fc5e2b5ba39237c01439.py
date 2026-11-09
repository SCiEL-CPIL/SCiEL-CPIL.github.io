"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 15
DEADLINE_MARGIN = 5.0  # seconds before 90s limit

def get_degrees(adj):
    """Return list of degrees for each vertex."""
    return [sum(adj[i]) for i in range(N)]

def is_valid(adj):
    """Check if no simple 3-edge path s-u-v-t has deg(s) == deg(t)."""
    deg = get_degrees(adj)
    for u in range(N):
        for v in range(u + 1, N):
            if not adj[u][v]:
                continue
            # For edge (u, v), find all s in N(u) and t in N(v) with s != t
            # and s != v, t != u (simple path)
            nu = [s for s in range(N) if adj[u][s] and s != v]
            nv = [t for t in range(N) if adj[v][t] and t != u]
            for s in nu:
                for t in nv:
                    if s == t:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def greedy_construct():
    """Greedy edge addition: try to add edges in a smart order."""
    adj = [[0] * N for _ in range(N)]
    # Start with a star (valid baseline)
    for v in range(1, N):
        adj[0][v] = 1
        adj[v][0] = 1
    
    # Generate all non-star edges
    remaining = [(u, v) for u in range(N) for v in range(u + 1, N) if not adj[u][v]]
    
    # Order by some heuristic: prefer edges that connect high-degree vertices?
    # Or just randomize and try
    random.shuffle(remaining)
    
    for u, v in remaining:
        adj[u][v] = 1
        adj[v][u] = 1
        if not is_valid(adj):
            adj[u][v] = 0
            adj[v][u] = 0
    
    return adj

def local_search(adj, max_iters=1000):
    """Try to improve by adding/removing/swapping edges."""
    best_adj = [row[:] for row in adj]
    best_count = sum(sum(row) for row in best_adj) // 2
    
    for _ in range(max_iters):
        improved = False
        # Try adding a missing edge
        for u in range(N):
            for v in range(u + 1, N):
                if not best_adj[u][v]:
                    best_adj[u][v] = 1
                    best_adj[v][u] = 1
                    if is_valid(best_adj):
                        best_count += 1
                        improved = True
                        break
                    else:
                        best_adj[u][v] = 0
                        best_adj[v][u] = 0
            if improved:
                break
        if improved:
            continue
        
        # Try removing an edge and adding two others (2-opt style)
        # This is expensive; skip for now
        break
    
    return best_adj, best_count

def construct():
    start_time = time.time()
    
    # Try multiple random restarts with different orderings
    best_adj = None
    best_count = 0
    
    for restart in range(20):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        adj = greedy_construct()
        count = sum(sum(row) for row in adj) // 2
        
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
        
        # Local search
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        adj_ls, count_ls = local_search(adj, max_iters=500)
        if count_ls > best_count:
            best_count = count_ls
            best_adj = adj_ls
    
    # Ensure we have at least the star
    if best_adj is None:
        best_adj = [[0] * N for _ in range(N)]
        for v in range(1, N):
            best_adj[0][v] = 1
            best_adj[v][0] = 1
        best_count = N - 1
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u][v]:
                edges.append([int(u), int(v)])
    
    return edges
# EVOLVE-BLOCK-END
