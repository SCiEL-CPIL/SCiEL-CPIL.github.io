# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n = 31
    m = 91  # number of edges required
    q = 5
    
    # Build PG(2,5) points as 1D subspaces of GF(5)^3
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                inv = pow(vec[first_nonzero], q-2, q)
                vec = [(x * inv) % q for x in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    # There should be q^2 + q + 1 = 25 + 5 + 1 = 31 points
    if len(points) != 31:
        # Fallback
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        return [[i, j] for i, j in all_possible[:m]]
    
    # Polarity: a point p=(x,y,z) is perpendicular to q'=(x',y',z') if xx'+yy'+zz' = 0 mod 5
    # The polarity graph connects p and q if p is perpendicular to q (and p != q)
    
    def perp(p, qpt):
        return (p[0]*qpt[0] + p[1]*qpt[1] + p[2]*qpt[2]) % q == 0
    
    # Build adjacency for the full polarity graph
    adj = [[False]*n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if perp(points[i], points[j]):
                adj[i][j] = adj[j][i] = True
                edges.append((i, j))
    
    # Count edges in polarity graph
    polarity_edges = len(edges)
    # Should be 93 (each vertex has degree 6, so 31*6/2 = 93)
    
    if polarity_edges < m:
        # Fallback: add random edges
        random.seed(42)
        existing = set(edges)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for e in all_possible:
            if e not in existing:
                edges.append(e)
                if len(edges) == m:
                    break
        return [[u, v] for u, v in edges]
    
    # We need to remove (polarity_edges - m) edges to get exactly m edges
    # Choose which edges to remove to minimize C4s
    
    # Helper to count C4s in the current graph
    def count_c4(adj_mat):
        # Number of C4s = (1/8) * sum over all u,v of (A^2[u][v] - A[u][v]) * (A^2[v][u] - A[v][u])
        # But this counts each C4 8 times (4 rotations * 2 directions)
        # Actually, let's use a more direct method for accuracy
        # A C4 is u-v-w-x-u where u,v,w,x are distinct
        # Count = (1/8) * trace(A^4) - ... this is complex
        
        # Simpler: for each pair of non-adjacent vertices (u,v), the number of C4s containing edge (u,v) is...
        # Actually, let's just count all C4s by iterating over pairs of edges
        # A C4 is formed by two non-adjacent vertices u,v and two common neighbors w,x of u and v
        # where w and x are adjacent to each other.
        
        # Better approach:
        # For each pair of vertices (u,v), let c = |N(u) ∩ N(v)|
        # The number of paths of length 2 from u to v is c.
        # A C4 u-w-v-x-u requires w,x in N(u)∩N(v) and w~x.
        
        # Let's use the formula:
        # #C4 = (1/8) * sum_{u,v} (A^2[u,v] - A[u,v]) * (A^2[v,u] - A[v,u]) ... wait, this is for counting cycles of length 4 in a different way.
        
        # Correct formula for #C4 in a simple graph:
        # #C4 = (1/8) * (sum_{i,j} (A^2[i,j])^2 - 2 * sum_{i,j} A[i,j]*A^2[i,j] - sum_{i,j} A^2[i,j])
        # This is getting messy. Let's use a direct enumeration for n=31 which is small enough.
        
        c4_count = 0
        # Iterate over all pairs of distinct vertices (u, v)
        for u in range(n):
            for v in range(u+1, n):
                # Find common neighbors
                common = []
                for w in range(n):
                    if w != u and w != v and adj_mat[u][w] and adj_mat[v][w]:
                        common.append(w)
                # For each pair of common neighbors (w, x), if w~x, then u-w-x-v-u is a C4
                # Wait, u-w-x-v-u is a C4 if w~x.
                # The cycle is u-w-x-v-u.
                # Vertices: u, w, x, v.
                # Edges: (u,w), (w,x), (x,v), (v,u).
                # We need (u,w), (w,x), (x,v), (v,u) to be edges.
                # We already have (u,w), (v,w) [since w is common], (v,x), (u,x) [since x is common].
                # So we just need (w,x) to be an edge.
                for i in range(len(common)):
                    for j in range(i+1, len(common)):
                        w = common[i]
                        x = common[j]
                        if adj_mat[w][x]:
                            c4_count += 1
        return c4_count
    
    # Helper to compute the number of C4s an edge (u,v) belongs to
    def count_c4s_for_edge(adj_mat, u, v):
        count = 0
        # A C4 containing edge (u,v) is of the form u-w-x-v-u
        # w is a neighbor of u (w != v)
        # x is a neighbor of v (x != u)
        # w and x are distinct
        # w and x are adjacent
        # Also w != x, w != u, w != v, x != u, x != v
        # Since w is neighbor of u, w != u.
        # Since x is neighbor of v, x != v.
        # We need w != v and x != u.
        # And w != x.
        # And adj[w][x] is True.
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj_mat[u][w]:
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    if adj_mat[v][x] and adj_mat[w][x]:
                        count += 1
        return count
    
    # Greedily remove edges with highest C4 count
    current_edges = set(edges)
    current_adj = [row[:] for row in adj]
    
    to_remove = polarity_edges - m
    removed = []
    
    for _ in range(to_remove):
        if time.time() > deadline:
            break
        # Find edge with max C4 count
        best_edge = None
        best_count = -1
        for e in current_edges:
            u, v = e
            # Recompute C4 count for this edge in current graph
            count = count_c4s_for_edge(current_adj, u, v)
            if count > best_count:
                best_count = count
                best_edge = e
        
        if best_edge is not None:
            current_edges.remove(best_edge)
            removed.append(best_edge)
            u, v = best_edge
            current_adj[u][v] = current_adj[v][u] = False
        else:
            break
    
    # Build final edge list
    final_edges = list(current_edges)
    
    # If we didn't remove enough edges (due to timeout), add random edges to reach m
    if len(final_edges) < m:
        existing = set(final_edges)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.seed(42)
        random.shuffle(all_possible)
        for e in all_possible:
            if e not in existing:
                final_edges.append(e)
                if len(final_edges) == m:
                    break
    
    # If we have too many (should not happen), trim
    if len(final_edges) > m:
        final_edges = final_edges[:m]
    
    # Verify we have exactly m edges
    if len(final_edges) != m:
        # Fallback
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        final_edges = all_possible[:m]
    
    # Verify all edges are distinct and valid
    for e in final_edges:
        u, v = e
        if not (0 <= u < n and 0 <= v < n and u != v):
            # Fallback
            random.seed(42)
            all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_possible)
            final_edges = all_possible[:m]
            break
    
    # Convert to list of lists
    result = [[u, v] for u, v in final_edges]
    return result

# EVOLVE-BLOCK-END
