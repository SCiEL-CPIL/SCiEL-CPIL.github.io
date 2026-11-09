import numpy as np
import itertools
import time

def construct():
    # Generate points of PG(2, 5)
    q = 5
    points = []
    # z = 1
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # z = 0, y = 1
    for x in range(q):
        points.append((x, 1, 0))
    # z = 0, y = 0, x = 1
    points.append((1, 0, 0))
    
    n = len(points) # 31
    
    # Define polarity matrix S
    # S = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    S = np.array([[0, 1, 0], 
                  [1, 0, 0], 
                  [0, 0, 1]], dtype=int)
    
    edges = []
    adj = [[False]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            p_i = np.array(points[i])
            p_j = np.array(points[j])
            # Check if p_i^T S p_j == 0 mod q
            val = (p_i @ (S @ p_j)) % q
            if val == 0:
                edges.append([i, j])
                adj[i][j] = True
                adj[j][i] = True
                
    # Count edges
    m = len(edges)
    
    # If m is not 93, we might have a different polarity or error.
    # Standard polarity of PG(2,5) should have 93 edges.
    
    # Function to count 4-cycles
    def count_4_cycles(adj):
        count = 0
        n = len(adj)
        # A 4-cycle is a set of 4 distinct vertices {a,b,c,d} such that
        # (a,b), (b,c), (c,d), (d,a) are edges.
        # To avoid double counting, we can iterate over pairs of edges (a,b) and (c,d) 
        # that share no vertices? No, that's for disjoint.
        # Standard way: For each pair of vertices (u, v), count common neighbors.
        # Let C(u, v) be the number of common neighbors of u and v.
        # The number of 4-cycles is (1/4) * sum_{u < v} C(u, v) * (C(u, v) - 1)?
        # No. 
        # Each 4-cycle u-v-w-x-u contributes 2 to the sum of common neighbors?
        # Let's stick to a direct enumeration for n=31, it's small enough.
        
        # Iterate over all pairs of edges (u,v) and (w,x) ?
        # Better: Iterate over all pairs of vertices (u, v).
        # Let N(u) be the set of neighbors of u.
        # For each pair (u, v), let k = |N(u) \cap N(v)|.
        # This counts the number of paths of length 2 between u and v.
        # A 4-cycle u-v-w-x-u consists of two paths of length 2 between u and w?
        # No, u and w are opposite.
        # The number of 4-cycles is (1/4) * sum_{u} sum_{v \in N(u)} (|N(u) \cap N(v)| - 1)?
        # Let's use the formula:
        # C4 = (1/4) * sum_{u < v} (C(u,v) choose 2) ? No.
        
        # Let's just enumerate all 4-tuples (a, b, c, d) with a < b < c < d.
        # Check if (a,b), (b,c), (c,d), (d,a) is a cycle? No, that's one specific ordering.
        # A set of 4 vertices forms a C4 if there are exactly 4 edges forming a cycle, or more.
        # The task asks for "exact number of four-cycles (subgraphs)".
        # This usually means the number of subgraphs isomorphic to C4.
        # In a K4, there are 3 such subgraphs.
        # In a graph, the number of C4 subgraphs is:
        # Sum over all pairs of opposite vertices (u, w) of (1/2) * (common_neighbors(u, w) choose 2)?
        # Let's verify.
        # For a fixed pair (u, w), let k be the number of common neighbors.
        # Any pair of common neighbors (v, x) forms a cycle u-v-w-x-u.
        # This is a C4.
        # Does this count each C4 exactly twice?
        # A C4 has two pairs of opposite vertices.
        # So summing over all pairs (u, w) and choosing 2 common neighbors counts each C4 twice.
        # So C4_count = (1/2) * sum_{u < w} (k_{uw} choose 2).
        
        total = 0
        for u in range(n):
            for w in range(u+1, n):
                # Count common neighbors
                common = 0
                for v in range(n):
                    if adj[u][v] and adj[w][v]:
                        common += 1
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    # Initial count
    c4_count = count_4_cycles(adj)
    
    # We need 91 edges.
    # If m > 91, remove edges.
    # If m < 91, we have a problem (should not happen).
    
    target_edges = 91
    current_edges = m
    
    if current_edges > target_edges:
        # Remove edges that are in the most 4-cycles
        # We need to identify which edges to remove.
        # For each edge (u, v), count how many 4-cycles it belongs to.
        # An edge (u, v) is in a 4-cycle u-v-w-x-u if w and x are common neighbors of u and v?
        # No. u-v is an edge. The cycle is u-v-w-x-u.
        # w is a neighbor of v, x is a neighbor of u.
        # And w-x is an edge.
        # So for each edge (u, v), the number of 4-cycles containing it is the number of pairs (w, x)
        # such that w \in N(v), x \in N(u), w \ne x, w \ne u, x \ne v, and (w, x) is an edge.
        
        def count_c4s_for_edge(u, v, adj):
            count = 0
            # w must be neighbor of v (and not u)
            # x must be neighbor of u (and not v)
            # and (w, x) must be an edge
            for w in range(n):
                if w == u or w == v: continue
                if adj[v][w]:
                    for x in range(n):
                        if x == u or x == v or x == w: continue
                        if adj[u][x] and adj[w][x]:
                            count += 1
            return count
        
        edges_to_remove = current_edges - target_edges
        edge_list = edges[:] # list of [u, v]
        
        for _ in range(edges_to_remove):
            # Find edge with max c4 count
            max_count = -1
            max_idx = -1
            for idx, e in enumerate(edge_list):
                u, v = e
                c = count_c4s_for_edge(u, v, adj)
                if c > max_count:
                    max_count = c
                    max_idx = idx
            
            if max_idx != -1:
                u, v = edge_list[max_idx]
                # Remove edge
                adj[u][v] = False
                adj[v][u] = False
                edge_list.pop(max_idx)
                
    return edge_list

