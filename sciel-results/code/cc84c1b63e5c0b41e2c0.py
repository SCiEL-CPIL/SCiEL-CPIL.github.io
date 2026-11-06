import itertools
import numpy as np

def construct():
    # 1. Generate all points in PG(2, 5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                inv_map = {1: 1, 2: 3, 3: 2, 4: 4}
                vec = (x, y, z)
                idx = 0
                if vec[0] == 0:
                    idx = 1
                if vec[0] == 0 and vec[1] == 0:
                    idx = 2
                val = vec[idx]
                inv_val = inv_map[val]
                c = inv_val
                cx = (x * c) % 5
                cy = (y * c) % 5
                cz = (z * c) % 5
                canon = (cx, cy, cz)
                if canon not in seen:
                    seen.add(canon)
                    points.append(canon)
    
    point_to_idx = {p: i for i, p in enumerate(points)}
    n_vertices = len(points)
    
    # 2. Construct the polarity graph
    adj = [[False] * n_vertices for _ in range(n_vertices)]
    edges = []
    
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if dot == 0:
                adj[i][j] = True
                adj[j][i] = True
                edges.append([i, j])
                
    M = len(edges)
    target_edges = 91
    
    # Helper to count C4s
    def count_c4s(adj_mat, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj_mat[i][j]:
                    # Count common neighbors
                    common = 0
                    for k in range(n):
                        if adj_mat[i][k] and adj_mat[j][k]:
                            common += 1
                    count += common
        return count // 2 # Each C4 counted twice (once for each pair of opposite vertices)
        
    # Actually, the number of C4s is sum over pairs (i,j) of |N(i) intersect N(j)| / 2?
    # No, sum |N(i) intersect N(j)| counts each C4 twice (i,j) and (k,l)?
    # A C4 is i-k-j-l-i.
    # Pairs of opposite vertices: (i,j) and (k,l).
    # For pair (i,j), k and l are in N(i) cap N(j).
    # So sum over all pairs (i,j) of |N(i) cap N(j)| = 2 * (# of C4s).
    # So count = sum / 2.

    if M > target_edges:
        # Remove M - target_edges edges
        num_remove = M - target_edges
        # Calculate C4 contribution for each edge
        # c_e = number of C4s containing e
        # This is hard to compute directly without listing C4s.
        # Approximation: Remove edges with high degree? Or just random?
        # Better: Greedy removal based on local C4 density.
        # For each edge (u,v), C4s involving it are determined by paths of length 2 between u and v.
        # c_e = |N(u) cap N(v)| ? No, that's paths of length 2.
        # A C4 containing (u,v) is u-v-a-b-u.
        # a in N(v), b in N(u), (a,b) edge.
        # So c_e is the number of edges between N(u) and N(v) (excluding u,v).
        
        edge_c4_count = {}
        for e in edges:
            u, v = e
            count = 0
            for a in range(n_vertices):
                if adj[u][a] and a != v:
                    for b in range(n_vertices):
                        if adj[v][b] and b != u and a != b:
                            if adj[a][b]:
                                count += 1
            edge_c4_count[e] = count
            
        # Sort edges by c4 count descending
        sorted_edges = sorted(edge_c4_count.items(), key=lambda x: x[1], reverse=True)
        
        # Remove top num_remove edges
        edges_to_remove = set()
        for e, _ in sorted_edges[:num_remove]:
            edges_to_remove.add(tuple(e))
            
        new_edges = []
        for e in edges:
            if tuple(e) not in edges_to_remove:
                new_edges.append(e)
                adj[e[0]][e[1]] = False
                adj[e[1]][e[0]] = False
            else:
                pass
                
        return new_edges
        
    elif M < target_edges:
        # Add target_edges - M edges
        num_add = target_edges - M
        # Find best edges to add
        # Similar to original logic
        
        def count_c4s_added(u, v):
            neighbors_u = [a for a in range(n_vertices) if adj[u][a]]
            neighbors_v = [b for b in range(n_vertices) if adj[v][b]]
            count = 0
            for a in neighbors_u:
                for b in neighbors_v:
                    if a != b and adj[a][b]:
                        count += 1
            return count

        best_edges = []
        min_c4s = float('inf')
        
        # We need to add num_add edges.
        # Greedy: add one by one, updating adj.
        
        for _ in range(num_add):
            best_edge = None
            min_c4s = float('inf')
            for i in range(n_vertices):
                for j in range(i + 1, n_vertices):
                    if not adj[i][j]:
                        c4_count = count_c4s_added(i, j)
                        if c4_count < min_c4s:
                            min_c4s = c4_count
                            best_edge = [i, j]
                            
            if best_edge:
                u, v = best_edge
                edges.append(best_edge)
                adj[u][v] = True
                adj[v][u] = True
                
        return edges
    else:
        return edges
