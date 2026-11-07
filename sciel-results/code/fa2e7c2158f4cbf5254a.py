"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    n = P['n']
    base_edges_set = set()
    for u, v in P['edges']:
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # 1. Identify Candidate Edges
    # To remain triangle-free on a path (which is bipartite), we must only add edges
    # between vertices of opposite parity. This ensures the graph remains bipartite.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 == 1 and (u, v) not in base_edges_set:
                candidates.append((u, v))
                
    # 2. Identify Critical Pairs
    # Pairs (u, v) such that distance in base path > 4.
    # In a path 0-...-23, dist(u,v) = |u-v|.
    # So critical pairs are all pairs with |u-v| > 4.
    critical_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if abs(u - v) > 4:
                critical_pairs.append((u, v))
                
    if not critical_pairs:
        return []

    # 3. Pre-compute "Coverage" for each candidate edge
    # An edge (a, b) "covers" a critical pair (u, v) if adding (a, b) reduces 
    # the shortest path distance between u and v to <= 4.
    # 
    # Distance in augmented graph <= 4 means there is a path of length <= 4.
    # Since we are adding edges to a path, the new path for (u,v) can be:
    # 1. Direct edge (if (u,v) is the candidate) -> dist 1
    # 2. u - ... - a - b - ... - v (using the new edge once)
    # 3. u - ... - a - b - ... - c - d - ... - v (using two new edges)
    # 4. u - ... - a - b - ... - c - d - ... - e - f - ... - v (using three new edges)
    #
    # This combinatorial check is expensive to do on-the-fly for Set Cover.
    # However, for n=24, the number of critical pairs is ~250.
    # Number of candidates is ~150.
    # 
    # Let's use a more robust approach: 
    # Since we want minimum edges, let's assume the optimal solution is small (3 or 4).
    # We can use a heuristic search (Greedy Set Cover) followed by Local Search (Swap/Remove).
    
    # Helper: Check if a set of edges makes diameter <= 4
    def is_valid(edge_set):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # BFS from each node to check diameter
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        q.append(v)
            # If any dist is -1, graph is disconnected (shouldn't happen if base is connected and we only add)
            # But let's check max dist <= 4
            if max(dist) > 4:
                return False
        return True

    # Helper: Greedy Set Cover
    # We approximate the "coverage" of an edge by how many critical pairs it "helps" directly.
    # A pair (u,v) is "fixed" by edge (a,b) if dist_path(u,v) using (a,b) <= 4.
    # dist_path(u,v) via (a,b) = min( dist_path(u,a) + 1 + dist_path(b,v), dist_path(u,b) + 1 + dist_path(a,v) )
    # where dist_path is distance in the BASE graph (path).
    # Note: This is an approximation because it assumes we don't use other new edges.
    # But for an initial seed, it's good.
    
    def base_dist(u, v):
        return abs(u - v)

    def potential_coverage(edge, critical_pairs):
        a, b = edge
        count = 0
        for u, v in critical_pairs:
            # Check if using this single edge reduces distance to <= 4
            d1 = base_dist(u, a) + 1 + base_dist(b, v)
            d2 = base_dist(u, b) + 1 + base_dist(a, v)
            if min(d1, d2) <= 4:
                count += 1
        return count

    # Rank candidates by potential coverage
    scored_candidates = []
    for e in candidates:
        score = potential_coverage(e, critical_pairs)
        if score > 0:
            scored_candidates.append((score, e))
            
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Greedy Selection
    selected_edges = set()
    remaining_pairs = set(critical_pairs)
    
    # We want to cover all remaining_pairs.
    # Since the "potential_coverage" is an overestimate (doesn't account for combined effects),
    # we might select too many. We will start with a greedy selection and then prune.
    
    # To be safe and fast, let's just pick the top K candidates that seem useful, 
    # then try to find a minimal subset.
    
    # Let's try k = 3, 4, 5.
    # We will generate a pool of "promising" edges.
    # Top 10 by score, plus some random/diverse ones?
    # Let's take top 15 by score.
    pool = [e for _, e in scored_candidates[:15]]
    
    # If pool is empty, something is wrong, but base path dist > 4 always exists for n=24.
    
    best_solution = None
    
    # Try to find a valid solution with size 3, then 4, etc.
    # We will search combinations from the pool.
    # C(15, 3) = 455. C(15, 4) = 1365. C(15, 5) = 3003.
    # This is very fast.
    
    for k in range(3, 8): # Try 3 to 7
        if time.time() > deadline:
            break
            
        # Generate combinations from the pool
        # To ensure diversity if top 15 fails, we could expand the pool in later iterations.
        # But let's stick to top 15 for now. If it fails, we expand.
        
        found = False
        for combo in itertools.combinations(pool, k):
            if time.time() > deadline:
                break
                
            if is_valid(combo):
                best_solution = list(combo)
                found = True
                break
                
        if found:
            break
            
    if best_solution is None:
        # If top 15 didn't work with up to 7 edges, expand pool and try again?
        # Or just return a known valid heuristic solution.
        # Let's try a simple heuristic: pick edges (0, 12), (6, 18), (12, 24->23?)
        # Actually, let's just run the greedy add/remove loop on the full candidate list as fallback.
        
        current_edges = set()
        # Add top 5 from pool
        for i in range(min(5, len(pool))):
            current_edges.add(pool[i])
            
        # Try to remove edges
        for e in list(current_edges):
            temp = set(current_edges)
            temp.remove(e)
            if is_valid(temp):
                current_edges = temp
        best_solution = list(current_edges)
        
    # Final check and cleanup
    # Ensure types are correct
    final_added = [[int(u), int(v)] for u, v in best_solution]
    
    # Sort for determinism? Not required but good.
    final_added.sort()
    
    return final_added

# EVOLVE-BLOCK-END
