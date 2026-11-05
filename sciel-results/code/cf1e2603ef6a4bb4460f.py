"""Polarity graph C4-free core + optimized extra edge + local search."""
import time
import random
from itertools import combinations

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
    # We use the standard representation: points are equivalence classes of non-zero vectors
    # Two vectors represent the same point if one is a scalar multiple of the other in F_5
    
    def normalize(v):
        """Normalize a vector in F_5^3 to a canonical representative."""
        # Find the first non-zero coordinate
        for i in range(3):
            if v[i] % 5 != 0:
                # Make this coordinate 1 by dividing by it (multiply by inverse)
                inv = pow(v[i] % 5, 3, 5)  # Fermat's little theorem: a^4 = 1 mod 5, so a^-1 = a^3 mod 5
                return [(v[j] * inv) % 5 for j in range(3)]
        return [0, 0, 0]
    
    points = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                norm = normalize([a, b, c])
                points.add(tuple(norm))
    
    points = list(points)
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Assign vertex indices to points
    point_to_vertex = {p: i for i, p in enumerate(points)}
    
    # Define the polarity: a symmetric bilinear form on F_5^3
    # Standard polarity for PG(2,5): use the symmetric matrix
    # [[1, 0, 0], [0, 1, 0], [0, 0, 1]] - this is just the dot product
    # But we need a polarity, which is a correlation of order 2
    # For q odd, a standard polarity is given by the symmetric bilinear form
    # with matrix J = [[0, 1, 0], [1, 0, 0], [0, 0, 1]] or similar
    
    # Let's use the standard polarity for the unitary/polarity graph construction
    # The polarity graph has edges between points x and y if x^T A y = 0
    # where A is a symmetric matrix defining the polarity
    
    # Use A = identity matrix for simplicity (this gives the standard polarity)
    # Actually, for a proper polarity, we need a symmetric bilinear form such that
    # the absolute points form a conic (for q odd)
    
    # Let's use A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] (identity)
    # Then x and y are orthogonal if x·y = 0 mod 5
    
    def are_orthogonal(p1, p2):
        """Check if two points are orthogonal under the polarity."""
        return (p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]) % 5 == 0
    
    # Build the polarity graph (C4-free core)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if are_orthogonal(points[i], points[j]):
                edges.add((i, j))
    
    # The polarity graph on PG(2,q) for q odd has (q+1)(q^2-1)/2 + q(q+1)/2 edges
    # Wait, let me recalculate. The number of edges in the polarity graph is:
    # Each point is orthogonal to q+1 other points (including itself if it's absolute)
    # For q odd, there are q+1 absolute points
    # Total edges = (number of ordered pairs (x,y) with x≠y and x⊥y) / 2
    
    # Actually, let's just count:
    num_edges_core = len(edges)
    # Expected: for PG(2,5), each point is orthogonal to 5+1=6 points (including itself if absolute)
    # Number of absolute points: q+1 = 6
    # Non-absolute points: 31-6 = 25, each orthogonal to 5 other points (not itself)
    # Absolute points: 6, each orthogonal to 5 other points (not itself)
    # Total ordered pairs: 6*5 + 25*5 = 30 + 125 = 155
    # But wait, orthogonality is symmetric, so unordered pairs = 155/2... that's not an integer
    # Let me recount: if x⊥y and x≠y, then y⊥x, so we count each pair twice in ordered pairs
    # Total ordered pairs with x⊥y, x≠y: for each x, count y≠x with x⊥y
    # For absolute x: x⊥x, so there are 6-1=5 other points y with x⊥y (since x is orthogonal to 6 total including itself)
    # For non-absolute x: x not⊥x, so there are 5 points y with x⊥y
    # Total: 6*5 + 25*5 = 155 ordered pairs
    # Unordered: 155/2 = 77.5... that's wrong
    
    # Let me just trust the computation. The polarity graph should be C4-free by construction.
    
    # We need 91 edges, so we need to add 91 - num_edges_core more edges
    if num_edges_core >= m:
        # If we already have enough edges, just take the first m
        edge_list = list(edges)[:m]
        return [[u, v] for u, v in edge_list]
    
    need_more = m - num_edges_core
    
    # Find the best extra edges to add (minimize C4 count)
    # For each candidate edge (u,v) not in the core, compute the number of C4s it would create
    # A C4 is created if u and v have common neighbors w1, w2 (distinct)
    # Number of new C4s = C(common_neighbors(u,v), 2)
    
    # Build adjacency for the core graph
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all candidate edges
    all_possible_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges:
                all_possible_edges.add((i, j))
    
    # Score each candidate edge by the number of C4s it would create
    def count_c4s_for_edge(u, v):
        """Count C4s created by adding edge (u,v)."""
        common = adj[u] & adj[v]
        return len(common) * (len(common) - 1) // 2
    
    # Sort candidates by C4 count
    candidates = []
    for u, v in all_possible_edges:
        c4_count = count_c4s_for_edge(u, v)
        candidates.append((c4_count, u, v))
    
    candidates.sort()
    
    # Add the best need_more edges
    added_edges = []
    for c4_count, u, v in candidates[:need_more]:
        edges.add((u, v))
        added_edges.append((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    # Now we have exactly m edges. Perform local search to reduce C4 count.
    
    def count_total_c4s(edge_set, adj_list):
        """Count total number of C4s in the graph."""
        count = 0
        for u in range(n):
            for v in adj_list[u]:
                if u < v:
                    common = adj_list[u] & adj_list[v]
                    count += len(common) * (len(common) - 1) // 2
        return count
    
    current_c4_count = count_total_c4s(edges, adj)
    
    # Local search: try swapping edges
    # For each edge in the graph, try removing it and adding a non-edge
    # Accept if C4 count decreases
    
    random.seed(42)
    
    # Precompute for speed
    def try_swap(edge_to_remove, edge_to_add, edge_set, adj_list):
        """Try swapping edge_to_remove with edge_to_add. Returns new C4 count or None if invalid."""
        u1, v1 = edge_to_remove
        u2, v2 = edge_to_add
        
        if edge_to_add in edge_set:
            return None
        if edge_to_remove not in edge_set:
            return None
        
        # Remove edge_to_remove
        edge_set.remove(edge_to_remove)
        adj_list[u1].remove(v1)
        adj_list[v1].remove(u1)
        
        # Add edge_to_add
        edge_set.add(edge_to_add)
        adj_list[u2].add(v2)
        adj_list[v2].add(u2)
        
        new_c4 = count_total_c4s(edge_set, adj_list)
        
        # Revert
        edge_set.remove(edge_to_add)
        edge_set.add(edge_to_remove)
        adj_list[u2].remove(v2)
        adj_list[v2].remove(u2)
        adj_list[u1].add(v1)
        adj_list[v1].add(u1)
        
        return new_c4
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Get current edges list
        current_edges = list(edges)
        random.shuffle(current_edges)
        
        for u1, v1 in current_edges:
            if time.time() > deadline - 5:
                break
            
            # Try adding a non-edge and removing this edge
            # Choose a random non-edge
            for _ in range(5):  # Try a few random non-edges
                if time.time() > deadline - 5:
                    break
                
                # Pick a random non-edge
                while True:
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 == v2:
                        continue
                    if u2 > v2:
                        u2, v2 = v2, u2
                    if (u2, v2) not in edges and (u2, v2) != (u1, v1):
                        break
                
                new_c4 = try_swap((u1, v1), (u2, v2), edges, adj)
                if new_c4 is not None and new_c4 < current_c4_count:
                    # Accept the swap
                    u1r, v1r = (u1, v1)
                    u2r, v2r = (u2, v2)
                    edges.remove((u1r, v1r))
                    adj[u1r].remove(v1r)
                    adj[v1r].remove(u1r)
                    edges.add((u2r, v2r))
                    adj[u2r].add(v2r)
                    adj[v2r].add(u2r)
                    current_c4_count = new_c4
                    improved = True
                    break
    
    # Final result
    return [[u, v] for u, v in edges]
