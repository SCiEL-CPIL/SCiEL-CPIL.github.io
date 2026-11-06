"""Optimized construction for fewest C4s on 31 vertices, 91 edges."""
import time
import numpy as np
from scipy.optimize import linprog
from itertools import combinations

N = 31
M = 91
DEADLINE = 110.0  # seconds, safety margin before 120s limit


def count_c4(edges):
    """Count the number of 4-cycles (subgraphs) in the graph defined by edges."""
    n = N
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # For each pair of vertices (u, v), count common neighbors.
    # A 4-cycle corresponds to two pairs of vertices (u,v) and (w,x) 
    # such that u-w, u-x, v-w, v-x are edges.
    # More efficiently: for each unordered pair {u,v}, let c(u,v) = |N(u) ∩ N(v)|.
    # The number of 4-cycles is (1/2) * sum over all unordered pairs {u,v} of C(c(u,v), 2).
    
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    common += 1
            total += common * (common - 1) // 2
    return total


def construct():
    start_time = time.time()
    
    # Strategy: Use a structured graph based on projective plane / polarity graph ideas.
    # n = 31 = q^2 + q + 1 with q = 5. This is the order of the projective plane PG(2,5).
    # The polarity graph of PG(2,5) has 31 vertices and is a known construction for C4-free or near-C4-free graphs.
    # However, ex(31, C4) = 90, so we need 91 edges. We'll start from a known good structure and add one edge.
    
    # Construct the polarity graph of PG(2,5).
    # Vertices are points of PG(2,5). Two points are adjacent if one lies on the polar line of the other.
    # Actually, let's use a simpler algebraic construction.
    
    # For q = 5, the Rees graph or a related construction can be used.
    # Let's use the standard polarity graph construction.
    
    # PG(2, q) has q^2 + q + 1 points. A polarity defines a self-dual correlation.
    # The polarity graph has vertices = points, and edges connect point P to point Q if Q lies on the polar line of P.
    # This graph is C4-free (or has few C4s) for prime powers q.
    
    # Let's build PG(2,5) points and a polarity.
    # Points of PG(2,5) are 1-dimensional subspaces of F_5^3.
    # We can represent them as normalized triples (x, y, z) where (x,y,z) != (0,0,0) and we fix a normalization.
    
    p = 5
    points = []
    for x in range(p):
        for y in range(p):
            for z in range(p):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make the first non-zero coordinate 1
                if x != 0:
                    points.append((1, y * pow(x, p-2, p) % p, z * pow(x, p-2, p) % p))
                elif y != 0:
                    points.append((0, 1, z * pow(y, p-2, p) % p))
                else:
                    points.append((0, 0, 1))
    
    # Should have p^2 + p + 1 = 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define a polarity. A standard polarity in PG(2,q) for odd q is given by the symmetric bilinear form:
    # B((x1,y1,z1), (x2,y2,z2)) = x1*x2 + y1*y2 + z1*z2 - x1*y2 - x2*y1
    # Wait, let's use a simpler one. The standard polarity for PG(2,q) when q is odd is:
    # (x1,y1,z1) is perpendicular to (x2,y2,z2) if x1*x2 + y1*y2 + z1*z2 + x1*y2 + x2*y1 = 0
    # Actually, let's use the well-known polarity: B(u,v) = u0*v0 + u1*v1 + u2*v2 + u0*v1 + u1*v0
    
    def perp(a, b):
        """Check if point a is perpendicular to point b under the polarity."""
        return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[0]*b[1] + a[1]*b[0]) % p == 0
    
    # Build the polarity graph
    edges_set = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if perp(points[i], points[j]):
                edges_set.add((points[i], points[j]))
    
    # Map points to indices 0..30
    point_to_idx = {pt: i for i, pt in enumerate(points)}
    
    polarity_edges = []
    for u_pt, v_pt in edges_set:
        u_idx = point_to_idx[u_pt]
        v_idx = point_to_idx[v_pt]
        polarity_edges.append((u_idx, v_idx))
    
    # The polarity graph of PG(2,q) for odd q has q^2 + q + 1 vertices and q(q+1)/2 * (q-1) + ... edges
    # Actually, each point has q+1 neighbors in the polarity graph (including itself if it's an absolute point).
    # For q=5, each vertex has degree q = 5 (if no absolute points) or q+1.
    # Total edges = 31 * 5 / 2 = 77.5... that's not integer. Let me recalculate.
    
    # In a polarity graph of PG(2,q) for odd q, each vertex is adjacent to q other vertices (not counting itself).
    # If there are absolute points, they are adjacent to themselves, but we only count distinct pairs.
    # Number of edges = (q^2 + q + 1) * q / 2
    # For q=5: 31 * 5 / 2 = 77.5. That's wrong.
    
    # Let me just count the edges we have.
    print(f"Polarity graph has {len(polarity_edges)} edges")
    
    # If we have fewer than 91 edges, we need to add more. If more, we need to remove some.
    # The goal is to have exactly 91 edges with minimum C4s.
    
    # Let's count C4s in the polarity graph
    current_edges = [list(e) for e in polarity_edges]
    current_c4 = count_c4(current_edges)
    print(f"Initial C4 count: {current_c4}")
    
    # If we have exactly 91 edges, we're done (modulo optimization)
    # If we have fewer, we add edges that minimize C4 increase
    # If we have more, we remove edges that minimize C4 decrease
    
    if len(current_edges) < M:
        # Add edges to reach M
        all_possible = set()
        for i in range(N):
            for j in range(i+1, N):
                all_possible.add((i, j))
        
        current_edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in current_edges)
        
        # Candidate edges to add
        candidates = [e for e in all_possible if e not in current_edge_set]
        
        # Sort by how many C4s they would create (approximation: count common neighbors of the pair)
        adj = [[False] * N for _ in range(N)]
        for u, v in current_edges:
            adj[u][v] = True
            adj[v][u] = True
        
        def c4_increase(edge):
            u, v = edge
            common = sum(1 for w in range(N) if adj[u][w] and adj[v][w])
            return common
        
        candidates.sort(key=c4_increase)
        
        # Add edges one by one, choosing the one with minimum C4 increase
        for _ in range(M - len(current_edges)):
            if time.time() - start_time > DEADLINE - 5:
                break
            if not candidates:
                break
            best_edge = None
            best_increase = float('inf')
            
            # Try a few candidates
            for cand in candidates[:50]:
                inc = c4_increase(cand)
                if inc < best_increase:
                    best_increase = inc
                    best_edge = cand
            
            if best_edge is None:
                best_edge = candidates[0]
            
            u, v = best_edge
            current_edges.append([u, v])
            current_edge_set.add((min(u, v), max(u, v)))
            adj[u][v] = True
            adj[v][u] = True
            candidates.remove(best_edge)
            
            # Update candidates' C4 increases (approximate: just remove ones involving u or v)
            candidates = [e for e in candidates if e[0] != u and e[1] != v and e[0] != v and e[1] != u]
    
    elif len(current_edges) > M:
        # Remove edges to reach M, choosing ones that reduce C4 the most
        current_c4 = count_c4(current_edges)
        
        for _ in range(len(current_edges) - M):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            best_edge = None
            best_c4_after = current_c4
            
            # Sample some edges to remove
            import random
            random.seed(42)
            sample = random.sample(range(len(current_edges)), min(50, len(current_edges)))
            
            for idx in sample:
                edge = current_edges[idx]
                # Remove this edge and count C4s
                test_edges = current_edges[:idx] + current_edges[idx+1:]
                c4_after = count_c4(test_edges)
                if c4_after < best_c4_after:
                    best_c4_after = c4_after
                    best_edge = idx
            
            if best_edge is not None:
                current_edges.pop(best_edge)
                current_c4 = best_c4_after
    
    # Final local optimization: try swapping edges to reduce C4s
    if time.time() - start_time < DEADLINE - 10:
        current_c4 = count_c4(current_edges)
        current_edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in current_edges)
        
        improved = True
        while improved and time.time() - start_time < DEADLINE - 10:
            improved = False
            # Try removing one edge and adding another
            for i in range(len(current_edges)):
                if time.time() - start_time > DEADLINE - 10:
                    break
                u1, v1 = current_edges[i]
                # Find candidate edges to add
                for u2 in range(N):
                    for v2 in range(u2+1, N):
                        if (u2, v2) in current_edge_set or (v2, u2) in current_edge_set:
                            continue
                        if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                            continue
                        # Try the swap
                        test_edges = current_edges[:i] + current_edges[i+1:] + [[u2, v2]]
                        c4_after = count_c4(test_edges)
                        if c4_after < current_c4:
                            current_edges[i] = [u2, v2]
                            current_edge_set.remove((min(u1, v1), max(u1, v1)))
                            current_edge_set.add((min(u2, v2), max(u2, v2)))
                            current_c4 = c4_after
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Ensure we have exactly M edges
    if len(current_edges) != M:
        # Fallback: just take first M from lexicographic order
        all_pairs = [[i, j] for i in range(N) for j in range(i+1, N)]
        current_edges = all_pairs[:M]
    
    return current_edges


if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
    print(f"Number of C4s: {count_c4(edges)}")
