"""Polarity graph C4-free core + optimized extra edge + local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Construct polarity graph on PG(2,5)
    # Points of PG(2,5) are 1D subspaces of F_5^3
    # There are (5^3 - 1)/(5 - 1) = 31 points
    # A polarity graph connects points if they are orthogonal under the polarity
    
    # Generate all 31 points in PG(2,5)
    # We represent points as vectors (a, b, c) with a normalization
    # To get unique representatives, we fix the first non-zero coordinate to 1
    
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = (a, b, c)
                # Find first non-zero
                if a != 0:
                    inv_a = pow(a, 3, 5)  # a^3 = a^-1 mod 5
                    norm = (1, (b * inv_a) % 5, (c * inv_a) % 5)
                elif b != 0:
                    inv_b = pow(b, 3, 5)
                    norm = (0, 1, (c * inv_b) % 5)
                else:  # c != 0
                    norm = (0, 0, 1)
                
                if norm not in seen:
                    seen.add(norm)
                    points.append(norm)
    
    # Should have exactly 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define polarity: A point p = (a,b,c) is orthogonal to q = (d,e,f) 
    # if a*d + b*e + c*f = 0 (mod 5) for the standard polarity
    # Actually, for a polarity graph, we need a symmetric bilinear form
    # Let's use the standard dot product mod 5
    
    # Build adjacency: edge between i and j if points[i] · points[j] = 0 mod 5
    # and i != j
    
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            pi = points[i]
            pj = points[j]
            dot = (pi[0] * pj[0] + pi[1] * pj[1] + pi[2] * pj[2]) % 5
            if dot == 0:
                adj[i, j] = adj[j, i] = True
    
    # Count edges in this polarity graph
    edge_count = int(np.sum(adj)) // 2
    
    # The polarity graph on PG(2, q) for odd q has q^2 + q + 1 vertices
    # and each vertex has degree q + 1 (for the standard polarity)
    # Wait, let me check: for PG(2, 5), q = 5
    # Number of edges should be (31 * 6) / 2 = 93? Let me verify.
    # Actually, in a polarity graph, the number of edges depends on the type
    # of the polarity. Let's just count what we have.
    
    # If we have exactly 90 edges, we need to add 1 more
    # If we have 91 edges, we're done
    # If we have more, we need to remove some
    
    # Let's see what we have
    if edge_count == 90:
        # Need to add exactly 1 edge
        # Find the non-edge that minimizes C4 count
        # A new edge (u,v) creates C4s equal to the number of common neighbors of u and v
        # that are not already connected... wait, no.
        
        # Actually, adding edge (u,v) creates a C4 for each pair of vertices (w,x) 
        # such that u-w-x-v-u forms a C4. This happens when:
        # w is adjacent to both u and v, and x is adjacent to both u and v, 
        # and w != x, and w-x is not an edge (to avoid triangles, but C4 doesn't require non-edges)
        # Wait, a C4 is u-w-x-v-u where all four edges exist.
        # So we need: u-w, w-x, x-v, v-u all exist.
        # Since we're adding v-u, we need u-w, w-x, x-v to exist in the original graph.
        # So the number of new C4s from adding (u,v) is the number of length-2 paths 
        # from u to v in the original graph, i.e., the number of common neighbors of u and v.
        
        # Wait, that's not quite right. A C4 through (u,v) is u-w-x-v-u.
        # This requires edges: u-w, w-x, x-v, v-u.
        # So w and x are common neighbors of u and v? No.
        # w is adjacent to u, x is adjacent to v, and w is adjacent to x.
        # So it's the number of edges (w,x) where w ~ u and x ~ v.
        
        # Actually, let me think again. The number of C4s containing a specific edge (u,v)
        # in a graph is equal to the number of paths of length 2 from u to v that don't use (u,v).
        # No wait. A C4 containing edge (u,v) is u-a-b-v-u where a-b is an edge.
        # So we need: a ~ u, b ~ v, and a ~ b.
        # The number of such C4s is the sum over all edges (a,b) of [a~u and b~v].
        # This is not simply the number of common neighbors.
        
        # Let me reconsider. When we add edge (u,v) to a C4-free graph, the number of new C4s
        # created is the number of length-2 paths from u to v in the original graph.
        # A length-2 path from u to v is u-w-v where w ~ u and w ~ v.
        # But that's a triangle, not a C4!
        
        # I'm confusing myself. Let me be precise.
        # A C4 is a cycle of length 4: u1-u2-u3-u4-u1.
        # If we add edge (u,v), new C4s are those that use this new edge.
        # So a new C4 is u-v-x-y-u where v-x, x-y, y-u are existing edges.
        # So we need: x ~ v, y ~ u, and x ~ y.
        # The number of such C4s is the number of edges (x,y) where x ~ v and y ~ u.
        
        # This can be computed as: for each neighbor x of v, for each neighbor y of u,
        # check if x ~ y. Count such pairs.
        
        # Alternatively, it's the inner product of the adjacency vectors of u and v,
        # but restricted to edges between them. Specifically, it's the number of edges 
        # in the bipartite graph between N(u) and N(v).
        
        # Let's compute this for each non-edge (u,v) and pick the one with minimum count.
        
        best_edge = None
        best_count = float('inf')
        
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    # Count edges between N(u) and N(v)
                    # N(u) = {w : adj[u,w]}
                    # N(v) = {w : adj[v,w]}
                    # Count pairs (x,y) with x in N(u), y in N(v), x ~ y
                    count = 0
                    for x in range(n):
                        if adj[u, x]:
                            for y in range(n):
                                if adj[v, y] and adj[x, y]:
                                    count += 1
                    # Each C4 is counted once (u-v-x-y-u), so count is the number of new C4s
                    if count < best_count:
                        best_count = count
                        best_edge = (u, v)
        
        # Add the best edge
        u, v = best_edge
        adj[u, v] = adj[v, u] = True
        
        # Now we have 91 edges. Let's do local search to minimize C4 count.
    
    elif edge_count == 91:
        # Already have 91 edges, do local search
        pass
    
    else:
        # Need to adjust to get exactly 91 edges
        # This case should not happen for a proper polarity graph, but let's handle it
        if edge_count > 91:
            # Remove edges to get to 91
            while int(np.sum(adj)) // 2 > 91:
                # Remove an edge that is in the most C4s
                # For now, just remove a random edge
                edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                random.shuffle(edges)
                u, v = edges[0]
                adj[u, v] = adj[v, u] = False
        else:
            # Add edges to get to 91
            while int(np.sum(adj)) // 2 < 91:
                # Add a random non-edge
                non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i, j]]
                random.shuffle(non_edges)
                u, v = non_edges[0]
                adj[u, v] = adj[v, u] = True
    
    # Count C4s in the current graph
    def count_c4s(adj_mat):
        # Number of C4s = sum over all pairs of vertices (u,v) of C(common_neighbors(u,v), 2)
        # where common_neighbors(u,v) is the number of vertices w adjacent to both u and v
        # But this counts each C4 twice (once for each pair of opposite vertices)
        # Actually, for a C4 u1-u2-u3-u4-u1, the pairs of opposite vertices are (u1,u3) and (u2,u4).
        # For pair (u1,u3), the common neighbors are u2 and u4, so C(2,2) = 1.
        # For pair (u2,u4), the common neighbors are u1 and u3, so C(2,2) = 1.
        # So each C4 is counted twice in the sum.
        # Therefore, number of C4s = (1/2) * sum_{u<v} C(common(u,v), 2)
        
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors
                common = np.sum(adj_mat[u] & adj_mat[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2
    
    # Local search: try edge swaps to reduce C4 count
    def try_swap(adj_mat, u1, v1, u2, v2, deadline):
        """Try swapping edges (u1,v1) and (u2,v2). Return new adj if it improves, else None."""
        if time.time() > deadline:
            return None
        
        # Check validity
        if not adj_mat[u1, v1] or not adj_mat[u2, v2]:
            return None
        if adj_mat[u1, u2] or adj_mat[u1, v2] or adj_mat[v1, u2] or adj_mat[v1, v2]:
            return None
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return None
        
        # Compute change in C4 count
        # This is complex; let's just do the swap and count
        old_adj = adj_mat.copy()
        adj_mat[u1, v1] = adj_mat[v1, u1] = False
        adj_mat[u2, v2] = adj_mat[v2, u2] = False
        adj_mat[u1, u2] = adj_mat[u2, u1] = True
        adj_mat[v1, v2] = adj_mat[v2, v1] = True
        
        new_count = count_c4s(adj_mat)
        old_count = count_c4s(old_adj)
        
        if new_count < old_count:
            return adj_mat
        else:
            return old_adj
    
    # Simple local search
    current_adj = adj.copy()
    current_count = count_c4s(current_adj)
    
    # Try multiple restarts with random perturbations
    best_adj = current_adj.copy()
    best_count = current_count
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Start from best or random perturbation
        if restart == 0:
            current_adj = best_adj.copy()
        else:
            current_adj = best_adj.copy()
            # Random perturbation: swap k random edges
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if current_adj[i, j]]
            random.shuffle(edges)
            for _ in range(3):
                u1, v1 = edges[random.randint(0, len(edges) - 1)]
                # Find a non-edge to swap with
                non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not current_adj[i, j]]
                random.shuffle(non_edges)
                u2, v2 = non_edges[random.randint(0, len(non_edges) - 1)]
                # Do the swap
                current_adj[u1, v1] = current_adj[v1, u1] = False
                current_adj[u2, v2] = current_adj[v2, u2] = False
                current_adj[u1, u2] = current_adj[u2, u1] = True
                current_adj[v1, v2] = current_adj[v2, v1] = True
        
        current_count = count_c4s(current_adj)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try all possible swaps
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if current_adj[i, j]]
            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not current_adj[i, j]]
            
            random.shuffle(edges)
            random.shuffle(non_edges)
            
            for u1, v1 in edges:
                if time.time() > deadline - 5:
                    break
                for u2, v2 in non_edges:
                    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                        continue
                    if current_adj[u1, u2] or current_adj[u1, v2] or current_adj[v1, u2] or current_adj[v1, v2]:
                        continue
                    
                    # Try the swap
                    new_adj = current_adj.copy()
                    new_adj[u1, v1] = new_adj[v1, u1] = False
                    new_adj[u2, v2] = new_adj[v2, u2] = False
                    new_adj[u1, u2] = new_adj[u2, u1] = True
                    new_adj[v1, v2] = new_adj[v2, v1] = True
                    
                    new_count = count_c4s(new_adj)
                    if new_count < current_count:
                        current_adj = new_adj
                        current_count = new_count
                        improved = True
                        break
                if improved:
                    break
        
        if current_count < best_count:
            best_count = current_count
            best_adj = current_adj.copy()
    
    # Convert to list of edges
    result_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                result_edges.append([i, j])
    
    return result_edges
