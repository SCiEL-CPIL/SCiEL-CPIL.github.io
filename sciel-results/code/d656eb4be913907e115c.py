"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _count_c4(adj, n):
    """Count 4-cycles in graph given adjacency matrix (n x n, 0/1)."""
    # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da
    # Count via: for each pair (u,v), count common neighbors. Each 4-cycle is counted once per pair of opposite vertices.
    # Actually: number of 4-cycles = (1/4) * sum over all unordered pairs {u,v} of (common_neighbors(u,v) choose 2)
    # Wait, that's not quite right. Let me think again.
    # A 4-cycle u-v-w-x-u: the pairs of opposite vertices are {u,w} and {v,x}.
    # For pair {u,w}, the common neighbors include v and x (and possibly others).
    # So if we sum over all pairs {u,w} of C(common(u,w), 2), each 4-cycle is counted exactly twice (once for each diagonal pair).
    # So # 4-cycles = (1/2) * sum_{u<w} C(cn(u,w), 2)
    
    c4_count = 0
    for u in range(n):
        for w in range(u+1, n):
            # common neighbors of u and w
            cn = np.sum(adj[u] & adj[w])
            if cn >= 2:
                c4_count += cn * (cn - 1) // 2
    return c4_count // 2

def _build_polarity_graph(n=31, q=5):
    """
    Build the polarity graph of the projective plane of order q=5.
    Vertices: 0..30 representing points of PG(2,5).
    Edge between u and v if point u lies on line v (i.e., u · v = 0 in GF(5)^3).
    This graph has n = q^2+q+1 = 31 vertices, and each vertex has degree q+1 = 6.
    Total edges = 31 * 6 / 2 = 93. But we need exactly 91.
    
    Actually, the polarity graph of PG(2,q) for odd q has n = q^2+q+1 vertices,
    each with degree q+1, so total edges = n*(q+1)/2 = 31*6/2 = 93.
    This is 2 more than ex(31;C4)=90.
    
    We'll construct this 93-edge graph, then remove 2 edges to get 91.
    """
    # Represent points of PG(2,5) as homogeneous coordinates [x:y:z] with not all zero,
    # modulo scalar multiples. We'll use a canonical representative.
    
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical form: make the first non-zero coordinate 1
                if x != 0:
                    inv_x = pow(x, -1, q)
                    pt = [1, (y * inv_x) % q, (z * inv_x) % q]
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    pt = [0, 1, (z * inv_y) % q]
                else:
                    pt = [0, 0, 1]
                if pt not in points:
                    points.append(pt)
    
    assert len(points) == n, f"Expected {n} points, got {len(points)}"
    
    # Build adjacency: u and v adjacent if points[u] dot points[v] == 0 mod q
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = sum(points[i][k] * points[j][k] for k in range(3)) % q
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append([i, j])
    
    return edges, adj

def _remove_edges_to_minimize_c4(edges, adj, n, num_to_remove=2, time_limit=100):
    """
    Given a graph with more edges than needed, remove edges to minimize C4 count.
    Use greedy + local search.
    """
    deadline = time.time() + time_limit
    current_edges = list(edges)
    current_adj = adj.copy()
    
    # Greedy: repeatedly remove the edge whose removal reduces C4 the most
    while len(current_edges) > 91 and time.time() < deadline:
        best_edge_idx = -1
        best_delta = 0  # delta in C4 count (negative means reduction)
        best_c4 = _count_c4(current_adj, n)
        
        for idx, (u, v) in enumerate(current_edges):
            # Try removing edge (u,v)
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            new_c4 = _count_c4(current_adj, n)
            delta = new_c4 - best_c4
            if delta < best_delta:
                best_delta = delta
                best_edge_idx = idx
            current_adj[u, v] = 1
            current_adj[v, u] = 1
        
        if best_edge_idx == -1:
            # Remove a random edge
            best_edge_idx = random.randint(0, len(current_edges)-1)
        
        # Remove the best edge
        u, v = current_edges[best_edge_idx]
        current_adj[u, v] = 0
        current_adj[v, u] = 0
        current_edges.pop(best_edge_idx)
    
    # Now we have 91 edges. Do local search: try swapping one edge for another
    # to reduce C4 count.
    current_c4 = _count_c4(current_adj, n)
    edge_set = set(tuple(sorted(e)) for e in current_edges)
    
    # Generate candidate non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = [p for p in all_pairs if p not in edge_set]
    
    best_c4 = current_c4
    best_edges = list(current_edges)
    best_adj = current_adj.copy()
    
    # Local search: try removing one edge and adding another
    while time.time() < deadline - 5:
        improved = False
        for _ in range(50):  # limit iterations per sweep
            if time.time() > deadline - 5:
                break
            # Pick a random edge to remove
            rem_idx = random.randint(0, len(best_edges)-1)
            u, v = best_edges[rem_idx]
            
            # Pick a random non-edge to add
            if not non_edges:
                break
            add_idx = random.randint(0, len(non_edges)-1)
            a, b = non_edges[add_idx]
            
            # Try the swap
            best_adj[u, v] = 0
            best_adj[v, u] = 0
            best_adj[a, b] = 1
            best_adj[b, a] = 1
            
            new_c4 = _count_c4(best_adj, n)
            
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_edges[rem_idx] = [a, b]
                edge_set.discard((u, v))
                edge_set.add((a, b))
                non_edges.remove((a, b))
                non_edges.append((u, v))
                improved = True
            else:
                # Revert
                best_adj[u, v] = 1
                best_adj[v, u] = 1
                best_adj[a, b] = 0
                best_adj[b, a] = 0
        
        if not improved:
            break
    
    return best_edges, best_c4

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build the polarity graph (93 edges)
    edges, adj = _build_polarity_graph(31, 5)
    
    # Remove 2 edges to get 91, minimizing C4
    result_edges, c4_count = _remove_edges_to_minimize_c4(
        edges, adj, 31, num_to_remove=2, time_limit=120
    )
    
    # Verify
    assert len(result_edges) == 91, f"Got {len(result_edges)} edges"
    assert len(set(tuple(sorted(e)) for e in result_edges)) == 91, "Duplicate edges"
    for e in result_edges:
        assert 0 <= e[0] <= 30 and 0 <= e[1] <= 30 and e[0] != e[1]
    
    return result_edges

# EVOLVE-BLOCK-END
