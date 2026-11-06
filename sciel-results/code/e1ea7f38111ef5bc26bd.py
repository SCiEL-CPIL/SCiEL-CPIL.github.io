import time
import random
import sys

def _check_c4_free(adj, n):
    # Check that no two vertices have 2 or more common neighbors
    for u in range(n):
        for v in range(u + 1, n):
            common = 0
            for w in adj[u]:
                if w in adj[v]:
                    common += 1
                    if common >= 2:
                        return False
    return True

def _independence_number(adj, n):
    # Exact independence number via branch and bound
    # Use bitsets for speed
    # adj is list of sets
    # We want to find max independent set
    
    # Precompute bitmasks
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    
    best = 0
    best_set = set()
    
    def dfs(candidates, current_set):
        nonlocal best, best_set
        if not candidates:
            if len(current_set) > best:
                best = len(current_set)
                best_set = set(current_set)
            return
        
        # If candidates + current_set <= best, prune
        if len(candidates) + len(current_set) <= best:
            return
            
        # Choose vertex with max degree in candidate subgraph (greedy heuristic)
        # Or just pick first one
        v = next(iter(candidates))
        
        # Option 1: Include v
        # New candidates are those not adjacent to v
        new_cands_incl = {u for u in candidates if u != v and u not in adj[v]}
        current_set.add(v)
        dfs(new_cands_incl, current_set)
        current_set.remove(v)
        
        # Option 2: Exclude v
        new_cands_excl = candidates - {v}
        dfs(new_cands_excl, current_set)
        
    dfs(set(range(n)), set())
    return best

def _verify_graph(n, edges):
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        adj[u].add(v)
        adj[v].add(u)
    
    # Check C4-free
    if not _check_c4_free(adj, n):
        return False
        
    # Check independence number < 11
    alpha = _independence_number(adj, n)
    if alpha >= 11:
        return False
        
    return True

def _generate_hs_graph():
    # Construct Hoffman-Singleton graph (50 vertices)
    # Vertices 0-48 correspond to (x, y) in F_7 x F_7
    # Vertex 49 is infinity
    # Adjacency:
    # 1. (x1, y1) ~ (x2, y2) if x1 != x2 and y1 - y2 = x1 * x2 (mod 7)
    # 2. (x, y) ~ infinity if y = x^2 (mod 7) ? No, standard construction is different.
    
    # Let's use a known explicit edge list for HS graph.
    # Source: House of Graphs or similar reliable source.
    # I will hardcode the edges.
    
    # The HS graph is the (50, 7, 0, 1) strongly regular graph.
    # Here is a verified edge list:
    edges = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
        [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14],
        [2, 15], [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21],
        [3, 22], [3, 23], [3, 24], [3, 25], [3, 26], [3, 27], [3, 28],
        [4, 29], [4, 30], [4, 31], [4, 32], [4, 33], [4, 34], [4, 35],
        [5, 36], [5, 37], [5, 38], [5, 39], [5, 40], [5, 41], [5, 42],
        [6, 43], [6, 44], [6, 45], [6, 46], [6, 47], [6, 48], [6, 49],
        [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14],
        [8, 15], [8, 16], [8, 17], [8, 18], [8, 19], [8, 20], [8, 21],
        [9, 22], [9, 23], [9, 24], [9, 25], [9, 26], [9, 27], [9, 28],
        [10, 29], [10, 30], [10, 31], [10, 32], [10, 33], [10, 34], [10, 35],
        [11, 36], [11, 37], [11, 38], [11, 39], [11, 40], [11, 41], [11, 42],
        [12, 43], [12, 44], [12, 45], [12, 46], [12, 47], [12, 48], [12, 49],
        [13, 15], [13, 16], [13, 17], [13, 18], [13, 19], [13, 20], [13, 21],
        [14, 22], [14, 23], [14, 24], [14, 25], [14, 26], [14, 27], [14, 28],
        [15, 29], [15, 30], [15, 31], [15, 32], [15, 33], [15, 34], [15, 35],
        [16, 36], [16, 37], [16, 38], [16, 39], [16, 40], [16, 41], [16, 42],
        [17, 43], [17, 44], [17, 45], [17, 46], [17, 47], [17, 48], [17, 49],
        [18, 22], [18, 23], [18, 24], [18, 25], [18, 26], [18, 27], [18, 28],
        [19, 29], [19, 30], [19, 31], [19, 32], [19, 33], [19, 34], [19, 35],
        [20, 36], [20, 37], [20, 38], [20, 39], [20, 40], [20, 41], [20, 42],
        [21, 43], [21, 44], [21, 45], [21, 46], [21, 47], [21, 48], [21, 49],
        [22, 29], [22, 30], [22, 31], [22, 32], [22, 33], [22, 34], [22, 35],
        [23, 36], [23, 37], [23, 38], [23, 39], [23, 40], [23, 41], [23, 42],
        [24, 43], [24, 44], [24, 45], [24, 46], [24, 47], [24, 48], [24, 49],
        [25, 15], [25, 16], [25, 17], [25, 18], [25, 19], [25, 20], [25, 21],
        [26, 22], [26, 23], [26, 24], [26, 25], [26, 27], [26, 28], # Wait, 26-25 already? No, 25-26 is same.
        # This hardcoded list is error-prone. I will use a programmatic construction instead.
        
        # Let's switch to the F_7 construction which is robust.
    ]
    
    # Robust F_7 Construction of HS Graph
    # Vertices: 0..48 are (x, y) in F_7 x F_7. 49 is infinity.
    # Two vertices (x1, y1) and (x2, y2) are adjacent if:
    # 1. x1 != x2 and y1 - y2 = x1 * x2 (mod 7)
    # 2. One is infinity and the other is (x, y) where y = x^2 (mod 7) ? 
    #    Actually, the standard construction for HS graph (50,7,0,1) is:
    #    Vertices are (x, y) in F_7^2 plus infinity.
    #    (x1, y1) ~ (x2, y2) iff x1 != x2 and y1 - y2 = x1 * x2 (mod 7).
    #    (x, y) ~ infinity iff y = x^2 (mod 7).
    #    Let's verify degree.
    #    For (x, y):
    #    - Neighbors in F_7^2: For each x' != x, there is a unique y' = y - x*x'. So 6 neighbors.
    #    - Neighbor at infinity: If y = x^2, then 1 neighbor. Else 0.
    #    This gives degree 6 or 7. Not regular. 
    #    
    #    Correct HS Graph Construction:
    #    It is the point graph of the generalized quadrangle GQ(3, 3).
    #    Or use the known edge list from a trusted source.
    
    # Since I cannot reliably generate HS edges from scratch without a reference, 
    # and the hardcoded list above is suspect, I will fall back to a greedy search 
    # that is guaranteed to be valid, aiming for n=39.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to use HS graph if we can verify it quickly
    # For now, let's use the greedy approach which is safer.
    
    # Greedy Construction
    # We build the graph vertex by vertex.
    # To ensure C4-free: When adding vertex v, it can only be connected to a set S of existing vertices
    # such that no two vertices in S have a common neighbor in the existing graph.
    # This means S must be an independent set in the "conflict graph" where u~v in conflict graph
    # if u and v have a common neighbor in G.
    
    # To ensure alpha(G) < 11:
    # We must ensure that the new vertex v does not create an independent set of size 11.
    # If v is connected to S, then v is NOT in any independent set containing any vertex in S.
    # v CAN be in an independent set if we pick vertices from V\S.
    # So alpha(G_new) = max( alpha(G_old), 1 + alpha(G_old[V\S]) ).
    # We need alpha(G_new) < 11.
    
    # Strategy:
    # 1. Start with empty graph.
    # 2. For i in 1..43:
    #    a. Compute conflict graph C of current G.
    #    b. Find Max Independent Set (MIS) in C. Let S be this set.
    #       Connect i to all vertices in S.
    #       This maximizes the degree of i while keeping C4-free.
    #    c. Check if this keeps alpha < 11.
    #       We need alpha(G[V\S]) < 10.
    #       If not, we need to choose a smaller S (remove some vertices from S) to increase alpha(G[V\S])?
    #       No, we want to DECREASE alpha(G[V\S])? 
    #       Wait. If S is large, V\S is small, so alpha(G[V\S]) is small.
    #       So connecting to a large S is good for keeping alpha low.
    #       So MIS in C is the best choice for S.
    #    d. If alpha(G_new) >= 11, we might need to backtrack or choose a different S.
    #       However, if we always choose max S, we are minimizing the remaining set V\S, which minimizes alpha.
    #       So if even with max S, alpha >= 11, then this vertex i cannot be added?
    #       Or we need to remove some edges from previous steps?
    
    # Let's implement this greedy.
    
    def greedy_build():
        n = 0
        adj = [set() for _ in range(43)] # Max 43
        edges = []
        
        for i in range(43):
            if time.time() > deadline:
                break
                
            # Compute conflict graph for current n vertices
            # Conflict[u][v] = 1 if u and v have common neighbor
            # We only care about vertices 0..n-1
            
            # Find MIS in conflict graph
            # Conflict graph vertices: 0..n-1
            # Edge (u,v) in conflict if |N(u) \cap N(v)| >= 1
            
            # Build adjacency for conflict graph
            conflict_adj = [set() for _ in range(n)]
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & adj[v]: # Common neighbor exists
                        conflict_adj[u].add(v)
                        conflict_adj[v].add(u)
                        
            # Find Max Independent Set in conflict_adj
            # Use a simple greedy or exact for small n
            # Since n <= 43, and conflict graph is likely dense, MIS is small.
            # Actually, if conflict graph is dense, MIS is small.
            # We want MAX independent set in conflict graph.
            
            # Greedy MIS:
            # Sort vertices by degree in conflict graph (ascending)
            # Pick vertex, remove it and its neighbors
            # This is a heuristic.
            
            # Better: Since we want to maximize S (neighbors of i), we want MIS of conflict graph.
            # Let's use a branch and bound for MIS, but it might be slow.
            # Given time limit, let's use a greedy heuristic first.
            
            # Heuristic: 
            # 1. Calculate degrees in conflict graph.
            # 2. Sort by degree.
            # 3. Iterate: pick lowest degree vertex, add to S, remove from conflict graph.
            
            remaining = set(range(n))
            S = set()
            
            while remaining:
                # Find vertex with min degree in remaining
                min_deg = float('inf')
                min_v = None
                for v in remaining:
                    deg = len(conflict_adj[v] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                        
                S.add(min_v)
                remaining.remove(min_v)
                remaining -= conflict_adj[min_v]
                
            # S is the set of neighbors for vertex i
            # Connect i to S
            for v in S:
                adj[i].add(v)
                adj[v].add(i)
                edges.append([i, v])
                
            # Check independence number
            # We need alpha(G) < 11
            # We can estimate or calculate exactly.
            # For speed, we can check if there is an independent set of size 11.
            # If not, we are good.
            
            # Simple check: 
            # If n is small, calculate exactly.
            # If n is large, use heuristic.
            
            # Let's calculate exact alpha for n <= 20, else heuristic.
            if n + 1 <= 20:
                alpha = _independence_number(adj, n + 1)
                if alpha >= 11:
                    # Rollback
                    for v in S:
                        adj[i].remove(v)
                        adj[v].remove(i)
                        edges.pop()
                    # Try to add i with no edges?
                    # Or skip i?
                    # If we skip i, we reduce n.
                    # Let's just skip i for now.
                    continue
            else:
                # Heuristic check
                # If we have many vertices, alpha is likely high.
                # We can check if the greedy independent set size is >= 11.
                # Greedy IS:
                rem = set(range(n + 1))
                is_set = set()
                while rem:
                    # Pick vertex with min degree in induced subgraph
                    min_deg = float('inf')
                    min_v = None
                    for v in rem:
                        deg = len(adj[v] & rem)
                        if deg < min_deg:
                            min_deg = deg
                            min_v = v
                    is_set.add(min_v)
                    rem.remove(min_v)
                    rem -= adj[min_v]
                    
                if len(is_set) >= 11:
                    # Rollback
                    for v in S:
                        adj[i].remove(v)
                        adj[v].remove(i)
                        edges.pop()
                    continue
                    
            n += 1
            
        return {'n': n, 'edges': edges}
        
    result = greedy_build()
    
    # Verify result
    if not _verify_graph(result['n'], result['edges']):
        # Fallback to seed
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
        
    return result
