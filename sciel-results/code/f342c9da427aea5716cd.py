import time
import sys
import random
import numpy as np
from collections import defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate the seed's LUW graph as a baseline
    n_luw, edges_luw = _construct_luw(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. Try Voltage Lift Construction
    # Base graph: K4 (4 vertices, 6 edges)
    # Lift size n = 4 * m
    # We want n < best_n
    max_m = (best_n - 1) // 4
    
    # Pre-compute non-backtracking closed walks in K4 up to length g-1
    # K4 vertices: 0, 1, 2, 3
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Index: 0:01, 1:02, 2:03, 3:12, 4:13, 5:23
    adj_k4 = {
        0: [(1, 0), (2, 1), (3, 2)],
        1: [(0, 0), (2, 3), (3, 4)],
        2: [(0, 1), (1, 3), (3, 5)],
        3: [(0, 2), (1, 4), (2, 5)]
    }
    
    walks = []
    # DFS to find non-backtracking closed walks
    # State: (current_node, prev_node, length, edge_sequence)
    # Start from each node
    for start in range(4):
        stack = [(start, -1, 0, [])]
        while stack:
            node, prev, length, seq = stack.pop()
            if length > 0 and node == start:
                # Found a closed walk
                # Check if it's non-backtracking (seq doesn't contain immediate reverse)
                # Since we build it non-backtracking, it is.
                # But we must ensure it's a "simple" cycle in terms of not just going back and forth?
                # Non-backtracking means we don't go u->v->u.
                # This ensures the walk is "elementary" in the sense of no immediate backtracks.
                # However, it might still visit nodes multiple times.
                # For girth check, we need to avoid ANY non-backtracking closed walk with sum 0.
                # Because if such a walk exists, the lift has a cycle of that length (or a divisor).
                # Actually, if a non-backtracking closed walk of length L has sum 0,
                # the lift contains a closed walk of length L.
                # Does it contain a simple cycle of length <= L?
                # Yes, any closed walk contains a simple cycle.
                # So we must avoid sum 0 for ALL non-backtracking closed walks of length < g.
                
                # Optimization: We only need to check walks up to length g-1.
                if length < g:
                    # Compute the linear form
                    # seq is list of (edge_idx, direction)
                    # direction: +1 if we traversed u->v where edge_idx is u-v (canonical), -1 if v->u
                    # We need to be careful with edge indexing.
                    # Let's store the coefficient vector.
                    coeff = [0] * 6
                    for edge_idx, dir_sign in seq:
                        coeff[edge_idx] += dir_sign
                    walks.append((length, tuple(coeff)))
            
            if length < g:
                for neighbor, edge_idx in adj_k4[node]:
                    if neighbor != prev:
                        # Determine direction
                        # Canonical edge is (min(u,v), max(u,v))
                        # If we are going from u to v:
                        # If u < v, we are going forward (+1)
                        # If u > v, we are going backward (-1)
                        # My adj_k4 stores (neighbor, edge_idx).
                        # I need to know the direction relative to the canonical order.
                        # Let's redefine adj_k4 to include direction info or compute it.
                        
                        # Let's just store the edge endpoints.
                        # Edge 0: 0-1. 0->1 is +1, 1->0 is -1.
                        # Edge 1: 0-2.
                        # Edge 2: 0-3.
                        # Edge 3: 1-2.
                        # Edge 4: 1-3.
                        # Edge 5: 2-3.
                        
                        u = node
                        v = neighbor
                        if u < v:
                            dir_sign = 1
                        else:
                            dir_sign = -1
                            
                        stack.append((neighbor, node, length + 1, seq + [(edge_idx, dir_sign)]))
    
    # Remove duplicates (same coeff vector)
    unique_walks = set()
    for length, coeff in walks:
        unique_walks.add(coeff)
    walk_coeffs = list(unique_walks)
    
    # Try to find labels for increasing m
    for m in range(128, max_m + 1):
        if time.time() > deadline - 20:
            break
            
        # Simulated Annealing to find labels x_0..x_5 in Z_m
        # Objective: Minimize number of walks with sum 0 mod m
        # If 0, we have a valid graph.
        
        # Initialize random labels
        labels = [random.randint(0, m-1) for _ in range(6)]
        
        def count_violations(labels):
            count = 0
            for coeff in walk_coeffs:
                s = 0
                for i, c in enumerate(coeff):
                    s += c * labels[i]
                if s % m == 0:
                    count += 1
            return count
        
        current_violations = count_violations(labels)
        if current_violations == 0:
            # Found a valid labeling!
            edges = _construct_lift(m, labels)
            if _has_girth_at_least_fast(4*m, edges, g):
                return 4*m, edges
        
        # Simple local search / SA
        temp = 1.0
        for step in range(5000):
            if time.time() > deadline - 20:
                break
                
            # Pick random edge label to change
            i = random.randint(0, 5)
            old_val = labels[i]
            new_val = random.randint(0, m-1)
            labels[i] = new_val
            
            new_violations = count_violations(labels)
            
            if new_violations < current_violations:
                current_violations = new_violations
                if current_violations == 0:
                    edges = _construct_lift(m, labels)
                    if _has_girth_at_least_fast(4*m, edges, g):
                        return 4*m, edges
            else:
                # Accept with probability exp(-(delta)/temp)
                delta = new_violations - current_violations
                if random.random() < np.exp(-delta / temp):
                    current_violations = new_violations
                else:
                    labels[i] = old_val
            
            temp *= 0.995
    
    return best_n, best_edges

def _construct_lift(m, labels):
    # Construct the lift graph
    # Vertices: (v, g) for v in 0..3, g in 0..m-1
    # Total vertices: 4*m
    n = 4 * m
    edges = set()
    
    # Edge definitions for K4
    # 0: 0-1, 1: 0-2, 2: 0-3, 3: 1-2, 4: 1-3, 5: 2-3
    k4_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    
    for i, (u, v) in enumerate(k4_edges):
        label = labels[i]
        for g in range(m):
            # Edge from (u, g) to (v, g+label)
            # And from (v, g-label) to (u, g) ?
            # The lift is undirected.
            # For each g, we add edge between (u, g) and (v, (g+label) % m)
            # This covers all edges in the lift.
            u_id = u * m + g
            v_id = v * m + (g + label) % m
            edges.add((u_id, v_id))
            
    return list(edges)

def _construct_luw(k, g):
    # Lazy-evaluated LUW construction from seed
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _random_3_reg_graph(n):
    """Generate a random simple 3-regular graph of n vertices using configuration model.
    Returns None if impossible or if multi-edges/loops are generated."""
    if n < 4 or n % 2 != 0:
        return None
    # Configuration model
    # Each vertex has 3 stubs
    stubs = [i for i in range(n) for _ in range(3)]
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None # Loop
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge
        edges.add((u, v))
    return list(edges)

def _has_girth_at_least_fast(n, edges, g):
    """Check if graph has no cycle shorter than g using NumPy matrix powers.
    For a bipartite graph (which we assume for even g), we check for cycles of length 4, 6, ..., g-2.
    A cycle of length 2k exists if the diagonal of A^(2k) is non-zero.
    """
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
        
    # We need to check for cycles of length 4, 6, ..., g-2
    # Since g is even, let g = 2m. We check 2, 4, ..., 2m-2.
    # Cycle of length 2 (multi-edge) is already excluded by simple graph check.
    # So we start from 4.
    
    # A^2
    # If any diagonal element of A^2 is > 0, there is a cycle of length 2 (impossible in simple graph) 
    # or we are checking for triangles? No, A^2[i,i] counts paths of length 2 from i to i.
    # In a simple graph, A^2[i,i] = degree(i). This is not a cycle.
    # A cycle of length L corresponds to closed walks of length L.
    # For simple graphs, the number of cycles of length L is related to trace(A^L).
    # Specifically, trace(A^L) counts closed walks.
    # For L=4, trace(A^4) counts closed walks of length 4.
    # A closed walk of length 4 in a simple graph is either a 4-cycle or a "back-and-forth" on two edges (i-j-i-k-i? No, i-j-i is length 2).
    # Actually, for simple graphs, if trace(A^L) > 0, it doesn't guarantee a simple cycle of length L, 
    # but for L=4, a closed walk i-j-k-j-i is length 4 but not a simple cycle.
    # However, if the graph has girth >= g, then there are NO closed walks of length < g that are not "backtracking".
    # This method is tricky.
    
    # Better: BFS is safer for small N.
    # But let's try the matrix method for speed.
    # Actually, for N=512, BFS is $512 \times 512 \approx 260k$ steps. 
    # In Python, 260k steps is ~0.1s. 
    # 5 attempts * 50 sizes = 250 checks * 0.1s = 25s. 
    # This fits in the budget!
    
    # So we can just use the standard BFS girth checker, but optimized.
    return _has_girth_at_least_bfs(n, edges, g)

def _has_girth_at_least_bfs(n, edges, g):
    """Standard BFS girth check, optimized with early exit."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # This cycle contains the start vertex.
                    # If this is the shortest cycle containing start, and we check all starts,
                    # we find the girth.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
