import time
import random
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer
    
    q = 5
    n = 31
    
    # 1. Generate 31 points of PG(2,5)
    points = []
    points.append((1, 0, 0))
    points.append((0, 1, 0))
    points.append((0, 0, 1))
    for y in range(1, q):
        points.append((1, y, 0))
    for z in range(1, q):
        points.append((1, 0, z))
    for z in range(1, q):
        points.append((0, 1, z))
    for y in range(1, q):
        for z in range(1, q):
            points.append((1, y, z))
            
    # 2. Construct Polarity Graph
    # Adjacency as list of sets for fast intersection
    adj = [set() for _ in range(n)]
    base_edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            if (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                base_edges.append((i, j))
                
    target_edges = 91
    edges_to_add = target_edges - len(base_edges) # Should be 1
    
    # If edges_to_add is not 1, we might need a more complex search.
    # But for q=5, it is exactly 1.
    
    if edges_to_add != 1:
        # Fallback to greedy + SA if the assumption breaks
        # But for this specific task, it should be 1.
        pass

    # Identify non-edges
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                non_edges.append((i, j))
                
    best_edge = None
    min_cost = float('inf')
    
    # Precompute neighbor sets for speed (already in adj)
    # For each non-edge (u, v), calculate the number of new C4s
    # Cost = number of edges (x, y) in G such that x in N(v) and y in N(u)
    
    for u, v in non_edges:
        A = adj[v]
        B = adj[u]
        
        # S = sum_{x in A} |N(x) intersect B|
        S = 0
        for x in A:
            S += len(adj[x] & B)
            
        C = A & B
        # E_CC = number of edges within C
        # Count degrees in induced subgraph on C
        deg_sum = 0
        for x in C:
            deg_sum += len(adj[x] & C)
        E_CC = deg_sum // 2
        
        cost = S - E_CC
        
        if cost < min_cost:
            min_cost = cost
            best_edge = (u, v)
            
    # If no non-edge found (should not happen), return base edges + any
    if best_edge is None:
        best_edge = non_edges[0] if non_edges else (0, 1)
        
    # Now try to improve by considering swaps: remove one base edge, add one non-edge
    # This could potentially reduce total C4s
    current_c4s = _count_c4s(adj, base_edges)
    best_solution = (current_c4s, base_edges.copy(), [best_edge])
    
    # Try local search: for each candidate added edge, try swapping with some removed edge
    # But this is expensive. Let's try a simpler approach: just evaluate all non-edges
    # and pick the one that minimizes total C4s
    
    # Actually, let's compute total C4s for each candidate edge
    best_total_c4s = float('inf')
    best_candidate = None
    
    for u, v in non_edges:
        if time.time() > deadline - 30:
            break
        # Add edge (u,v) and count C4s
        adj[u].add(v)
        adj[v].add(u)
        c4s = _count_c4s(adj, base_edges + [(u,v)])
        adj[u].discard(v)
        adj[v].discard(u)
        
        if c4s < best_total_c4s:
            best_total_c4s = c4s
            best_candidate = (u, v)
    
    if best_candidate is not None:
        best_edge = best_candidate
    
    # Now try swaps: remove one edge from base, add best_candidate
    # This is more expensive but could help
    if time.time() < deadline - 20:
        # Try a few random swaps
        random.seed(42)
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            # Pick a random base edge to remove
            idx = random.randint(0, len(base_edges) - 1)
            rem_u, rem_v = base_edges[idx]
            
            # Create modified adjacency
            adj2 = [set(s) for s in adj]
            adj2[rem_u].discard(rem_v)
            adj2[rem_v].discard(rem_u)
            
            # Add best_candidate edge
            add_u, add_v = best_edge
            adj2[add_u].add(add_v)
            adj2[add_v].add(add_u)
            
            # Count C4s
            c4s = _count_c4s(adj2, [e for i, e in enumerate(base_edges) if i != idx] + [(add_u, add_v)])
            
            if c4s < best_total_c4s:
                best_total_c4s = c4s
                # Record this swap
                # We need to track which edge was removed
                # For simplicity, just keep the best add edge for now
    
    final_edges = []
    for u, v in base_edges:
        final_edges.append([u, v])
    final_edges.append([best_edge[0], best_edge[1]])
        
    return final_edges

def _count_c4s(adj, edges):
    """Count the number of 4-cycles in the graph defined by adj and edges."""
    n = len(adj)
    c4_count = 0
    # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges (a,b), (b,c), (c,d), (d,a)
    # We can count by looking at pairs of vertices and their common neighbors
    # For each pair (u,v), count the number of common neighbors w
    # Then sum over all pairs, but we need to be careful about double counting
    
    # Better approach: for each vertex u, look at its neighbors
    # For each pair of neighbors (v,w) of u, count how many common neighbors they have
    # excluding u itself
    
    # Even better: use the standard formula
    # C4 = sum over all pairs (u,v) of C(common_neighbors(u,v), 2) / 4
    # Wait, that's not right either
    
    # Let's use: for each ordered pair (u,v) where u < v,
    # let cn(u,v) = |N(u) ∩ N(v)|
    # Then the number of 4-cycles containing edge (u,v) as a diagonal is...
    # Actually, a 4-cycle has 2 diagonals. Each 4-cycle is counted twice if we sum over all pairs.
    
    # Standard method:
    # For each vertex u, consider its neighbors N(u)
    # For each pair (v,w) in N(u) with v < w, the number of 4-cycles u-v-x-w-u
    # is |N(v) ∩ N(w)| - 1 (excluding u)
    # But this counts each 4-cycle 4 times (once for each vertex)
    
    # Simpler: just enumerate all 4-cycles directly
    # For each pair of non-adjacent vertices (u,v), count common neighbors
    # Each 4-cycle u-a-v-b-u contributes to common_neighbors(u,v) via a and b
    
    # Let's use the efficient method:
    # C4 = (1/4) * sum_{u} sum_{v,w in N(u), v<w} (|N(v) ∩ N(w)| - 1)
    # But this is still O(n * d^2 * d) which might be slow
    
    # For n=31, let's just do a direct enumeration
    # There are C(31,4) = 31465 ways to choose 4 vertices
    # For each, check if they form a 4-cycle
    
    c4_count = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Check if {a,b,c,d} forms a 4-cycle
                    # A 4-cycle has exactly 4 edges among these 4 vertices
                    # and they form a cycle
                    # The 4 vertices can form a cycle in 3 ways:
                    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
                    # But we need to check which edges exist
                    
                    # Count edges among {a,b,c,d}
                    edges_among = 0
                    if b in adj[a]: edges_among += 1
                    if c in adj[a]: edges_among += 1
                    if d in adj[a]: edges_among += 1
                    if c in adj[b]: edges_among += 1
                    if d in adj[b]: edges_among += 1
                    if d in adj[c]: edges_among += 1
                    
                    if edges_among == 4:
                        # Check if it's a cycle (not a path or other shape)
                        # With 4 edges on 4 vertices, it must be a cycle or a triangle with a pendant
                        # But a triangle with pendant has a vertex of degree 3
                        # Check degrees in the induced subgraph
                        deg_a = (1 if b in adj[a] else 0) + (1 if c in adj[a] else 0) + (1 if d in adj[a] else 0)
                        deg_b = (1 if a in adj[b] else 0) + (1 if c in adj[b] else 0) + (1 if d in adj[b] else 0)
                        deg_c = (1 if a in adj[c] else 0) + (1 if b in adj[c] else 0) + (1 if d in adj[c] else 0)
                        deg_d = (1 if a in adj[d] else 0) + (1 if b in adj[d] else 0) + (1 if c in adj[d] else 0)
                        
                        # A 4-cycle has all degrees = 2
                        if deg_a == 2 and deg_b == 2 and deg_c == 2 and deg_d == 2:
                            c4_count += 1
    
    return c4_count
