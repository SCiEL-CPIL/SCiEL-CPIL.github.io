"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import numpy as np

N = 31
E = 91
DEADLINE = 110.0  # safety margin before 120s limit

def _count_c4(adj):
    """Count the number of 4-cycles in the graph given by adjacency matrix."""
    # A 4-cycle is counted 8 times by (A^2)_{ij} for i<j (each C4 has 4 pairs of opposite vertices, 
    # and each such pair is connected via 2 paths of length 2, but we need to be careful).
    # Actually, the number of 4-cycles = (1/8) * sum over all unordered pairs {u,v} of (common_neighbors(u,v))^2 - ... 
    # Wait, no. Let's use the standard formula:
    # Number of C4 = (1/8) * trace(A^4) - ... no that counts walks.
    # Better: For each pair of vertices (u,v), let c(u,v) = number of common neighbors.
    # Then the number of 4-cycles is (1/8) * sum_{u < v} c(u,v) * (c(u,v) - 1) ... no.
    # Actually, each 4-cycle u-v-w-x-u contributes 1 to c(u,w) and 1 to c(v,x). 
    # The number of 4-cycles = (1/8) * sum over all unordered pairs {u,v} of c(u,v) * (c(u,v) - 1)
    # Wait, let me think again. A 4-cycle is u-v-w-x-u. The pairs of opposite vertices are {u,w} and {v,x}.
    # For pair {u,w}, the common neighbors include v and x (and possibly others). 
    # Each 4-cycle corresponds to choosing 2 pairs of opposite vertices. 
    # The number of 4-cycles = (1/8) * sum_{u<v} [c(u,v) choose 2]
    # Because for each pair of opposite vertices {u,v}, if they have c common neighbors, 
    # then each pair of common neighbors forms a 4-cycle. But each 4-cycle is counted 
    # exactly twice (once for each pair of opposite vertices). And each 4-cycle has 2 pairs 
    # of opposite vertices. So total = (1/2) * sum_{u<v} C(c(u,v), 2) = (1/8) * sum_{u<v} c(u,v)(c(u,v)-1).
    # Wait: sum_{u<v} C(c(u,v), 2) counts each 4-cycle exactly 2 times (once per pair of opposite vertices).
    # So #C4 = (1/2) * sum_{u<v} C(c(u,v), 2) = (1/8) * sum_{u<v} c(u,v)*(c(u,v)-1)
    
    # Compute common neighbors using matrix multiplication
    A = adj
    common = A @ A  # common[i][j] = number of common neighbors of i and j
    # Sum over i < j of C(common[i][j], 2)
    total = 0
    for i in range(N):
        for j in range(i+1, N):
            c = common[i][j]
            total += c * (c - 1) // 2
    return total // 2

def _build_polarity_graph():
    """
    Build the polarity graph of PG(2,5).
    PG(2,5) has 31 points. The polarity graph connects two points if they are 
    conjugate with respect to a polarity. This gives a C4-free graph with 90 edges.
    
    For q=5, we use the standard construction.
    Points of PG(2,5) can be represented as equivalence classes of non-zero vectors in GF(5)^3.
    There are (5^3 - 1)/(5-1) = 124/4 = 31 points.
    
    A polarity in PG(2,q) for odd q can be defined using a symmetric matrix.
    For q=5, we use the Hermitian polarity or a symplectic-type polarity.
    
    Actually, for the polarity graph of PG(2,q) with q odd, the graph has q^2+q+1 vertices 
    and each vertex has degree q. The total number of edges is q(q^2+q+1)/2 = q(q+1)(q+2)/6... 
    wait, q(q^2+q+1)/2 for q=5 is 5*31/2 = 77.5, not an integer. So degree is not q.
    
    Let me reconsider. The polarity graph of PG(2,q) for odd q:
    - Vertices: points of PG(2,q), so q^2+q+1 = 31
    - Two vertices are adjacent if they are conjugate (i.e., the line through them is tangent to the absolute points)
    - The graph is C4-free
    - Each vertex has degree q (for the standard polarity)
    - Total edges = q(q^2+q+1)/2... but q(q^2+q+1) must be even. For q=5: 5*31=155, odd. So degree is not q.
    
    Hmm, let me look this up more carefully. For the polarity graph of PG(2,q):
    - If q is odd, the graph has q^2+q+1 vertices, each of degree q, so total edges = q(q^2+q+1)/2.
    - But q(q^2+q+1) = 5*31 = 155, which is odd. Contradiction.
    
    Wait, I think the degree is q for q even, and something else for q odd. Let me just construct it directly.
    
    For PG(2,5), we can represent points as (x, y, z) in GF(5)^3 up to scalar, not all zero.
    A standard polarity is given by the symmetric bilinear form:
    B((x1,y1,z1), (x2,y2,z2)) = x1*x2 + y1*y2 + z1*z2
    But this is just the standard dot product. Two points are conjugate if B(p,q) = 0.
    
    Let's enumerate all 31 points and build the graph.
    """
    # Enumerate all projective points in PG(2,5)
    # Represent each point by a canonical form: first non-zero coordinate is 1
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical form: make first non-zero entry 1
                vec = [x, y, z]
                # Find first non-zero
                idx = -1
                for i in range(3):
                    if vec[i] != 0:
                        idx = i
                        break
                if idx == -1:
                    continue
                # Divide by vec[idx] mod 5
                inv = pow(vec[idx], 3, 5)  # Fermat's little theorem: a^(p-2) = a^(-1) mod p
                vec = [(v * inv) % 5 for v in vec]
                key = tuple(vec)
                if key not in points:
                    points.append(key)
    
    # Should have 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Build adjacency: two points are adjacent if their dot product is 0 mod 5
    # and they are not the same point
    adj = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i+1, N):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj[i][j] = 1
                adj[j][i] = 1
    
    # Count edges
    num_edges = int(adj.sum() // 2)
    
    # This should be C4-free with some number of edges. We need 91 edges.
    # If the polarity graph has fewer than 91 edges, we add more.
    # If it has more, we need to remove some.
    
    return adj, num_edges

def _add_min_edge(adj, edges_list):
    """Add the edge that creates the fewest new 4-cycles."""
    # For each non-edge, compute how many new C4s it would create
    # A new edge (u,v) creates a C4 for each pair of common neighbors (w,x) 
    # such that w-x is an edge... actually, adding edge (u,v) creates a C4 
    # u-w-v-x-u if w and x are common neighbors of u and v and w-x is an edge.
    # More precisely: adding (u,v) creates a C4 for each path of length 3 from u to v 
    # that doesn't use (u,v). That is, for each w, x such that u-w, w-x, x-v are edges.
    # The number of new C4s = number of length-3 paths from u to v.
    
    # Precompute: for each non-edge (u,v), count length-3 paths
    # A length-3 path u-w-x-v means w is neighbor of u, x is neighbor of v, and w-x is an edge.
    
    best_edge = None
    best_count = float('inf')
    
    existing = set()
    for u, v in edges_list:
        existing.add((min(u,v), max(u,v)))
    
    for u in range(N):
        for v in range(u+1, N):
            if (u, v) in existing:
                continue
            # Count length-3 paths from u to v
            count = 0
            for w in range(N):
                if adj[u][w] and w != v:
                    for x in range(N):
                        if adj[w][x] and adj[x][v] and x != u and w != x:
                            count += 1
            # Each C4 is counted... let's think. A C4 u-w-x-v-u is counted once for the 
            # specific w and x. But the same C4 can be traversed in different ways? 
            # No, a length-3 path u-w-x-v is uniquely determined by w and x.
            # But each C4 has 4 such paths (one for each starting vertex). Since we're 
            # counting paths from u to v, each C4 containing edge (u,v) is counted exactly 
            # once (the path u-...-v that goes the long way around).
            # Actually, a C4 is u-w-x-v-u. The length-3 path from u to v is u-w-x-v. 
            # There's only one such path for a given C4. So count = number of new C4s.
            
            if count < best_count:
                best_count = count
                best_edge = (u, v)
    
    return best_edge, best_count

def _local_search(adj, edges_list, time_limit):
    """Local search: try swapping edges to reduce C4 count."""
    start_time = time.time()
    
    # Get current C4 count
    current_c4 = _count_c4(adj)
    
    existing = set()
    for u, v in edges_list:
        existing.add((min(u,v), max(u,v)))
    
    # Try single edge swaps: remove edge (a,b), add edge (c,d)
    # This is expensive. Let's try simpler moves first.
    
    # Move type 1: Remove an edge and add a different edge (keeping 91 edges)
    # This changes the graph by 2 edges.
    
    # For efficiency, let's try a random restart approach with simulated annealing
    
    rng = np.random.default_rng(42)
    
    best_adj = adj.copy()
    best_edges = edges_list[:]
    best_c4 = current_c4
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Randomly pick an edge to remove
        idx = rng.integers(0, len(best_edges))
        remove_u, remove_v = best_edges[idx]
        
        # Randomly pick a non-edge to add
        while True:
            add_u = rng.integers(0, N)
            add_v = rng.integers(0, N)
            if add_u == add_v:
                continue
            key = (min(add_u, add_v), max(add_u, add_v))
            if key not in existing:
                break
        
        # Apply the swap
        new_adj = best_adj.copy()
        new_adj[remove_u][remove_v] = 0
        new_adj[remove_v][remove_u] = 0
        new_adj[add_u][add_v] = 1
        new_adj[add_v][add_u] = 1
        
        new_c4 = _count_c4(new_adj)
        
        delta = new_c4 - best_c4
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            # Accept
            best_adj = new_adj
            best_edges = best_edges[:]
            best_edges[idx] = [add_u, add_v]
            existing.remove((min(remove_u, remove_v), max(remove_u, remove_v)))
            existing.add(key)
            best_c4 = new_c4
        
        temp *= cooling
    
    return best_adj, best_edges, best_c4

def construct():
    start_time = time.time()
    
    # Build polarity graph
    adj, num_edges = _build_polarity_graph()
    
    # Convert to edge list
    edges_list = []
    for u in range(N):
        for v in range(u+1, N):
            if adj[u][v]:
                edges_list.append([u, v])
    
    current_edges = len(edges_list)
    
    # We need exactly 91 edges. The polarity graph might have a different number.
    # If we have fewer than 91, add edges that minimize C4s.
    # If we have more, remove edges that minimize C4 increase.
    
    if current_edges < E:
        # Add edges one by one, choosing the one that adds fewest C4s
        for _ in range(E - current_edges):
            edge, count = _add_min_edge(adj, edges_list)
            if edge is None:
                break
            u, v = edge
            adj[u][v] = 1
            adj[v][u] = 1
            edges_list.append([u, v])
    elif current_edges > E:
        # Remove edges one by one, choosing the one whose removal reduces C4s the most
        for _ in range(current_edges - E):
            best_remove = None
            best_delta = float('inf')
            for idx, (u, v) in enumerate(edges_list):
                # Temporarily remove
                adj[u][v] = 0
                adj[v][u] = 0
                new_c4 = _count_c4(adj)
                adj[u][v] = 1
                adj[v][u] = 1
                delta = new_c4 - _count_c4(adj)
                if delta < best_delta:
                    best_delta = delta
                    best_remove = idx
            if best_remove is not None:
                u, v = edges_list[best_remove]
                adj[u][v] = 0
                adj[v][u] = 0
                edges_list.pop(best_remove)
    
    # Now do local search with simulated annealing
    remaining_time = DEADLINE - (time.time() - start_time) - 10.0  # 10s safety margin
    if remaining_time > 5.0:
        adj, edges_list, final_c4 = _local_search(adj, edges_list, remaining_time)
    
    # Ensure we have exactly 91 edges
    assert len(edges_list) == E, f"Expected {E} edges, got {len(edges_list)}"
    
    # Verify all edges are distinct and valid
    edge_set = set()
    for u, v in edges_list:
        key = (min(u, v), max(u, v))
        assert 0 <= u < N and 0 <= v < N, f"Invalid vertex: {u}, {v}"
        assert u != v, f"Self-loop: {u}"
        assert key not in edge_set, f"Duplicate edge: {key}"
        edge_set.add(key)
    
    return edges_list
