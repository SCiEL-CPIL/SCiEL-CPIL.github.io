"""Voltage graph lift approach for cage problem n(4,10)."""
import numpy as np
from itertools import product
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using voltage graph lifting.
    
    Strategy: Use a small base graph (K_{4,4} for k=4) and lift it via a voltage
    assignment over a cyclic group to increase girth while keeping vertex count low.
    
    For k=4, g=10: We use K_{4,4} (8 vertices) and lift over Z_N.
    Target N ~ 40 to get ~320 vertices.
    """
    if k != 4 or g != 10:
        # Fallback: use Lazebnik-Ustimenko-Woldar construction
        return _luw_construct(k, g)
    
    # Try voltage lift of K_{4,4}
    # K_{4,4} vertices: 0..3 (U), 4..7 (W)
    # Edges: (i, 4+j) for i in 0..3, j in 0..3
    
    # We search for N and a voltage matrix M[4][4]
    # such that the lifted graph has girth >= 10.
    
    # We try N from 40 upwards.
    # We use random voltage matrices.
    
    max_n_tries = 5000
    current_time = 0
    import time
    start_time = time.time()
    timeout = 110 # Leave 10s for overhead
    
    for N in range(40, 100):
        if time.time() - start_time > timeout:
            break
            
        # Try random matrices
        for _ in range(max_n_tries // 10):
            if time.time() - start_time > timeout:
                break
            
            # Generate random M
            M = np.random.randint(0, N, size=(4, 4))
            
            # Check 4-cycles first (fast)
            # Condition: M[i][j] + M[k][l] != M[i][l] + M[k][j] mod N
            # for all i != k, j != l
            ok_4 = True
            for i in range(4):
                for k in range(i+1, 4):
                    for j in range(4):
                        for l in range(j+1, 4):
                            if (M[i][j] + M[k][l] - M[i][l] - M[k][j]) % N == 0:
                                ok_4 = False
                                break
                        if not ok_4: break
                    if not ok_4: break
                if not ok_4: break
            
            if not ok_4:
                continue
            
            # Construct graph and check girth
            n, edges = _build_voltage_graph(N, M)
            if _has_girth_at_least(n, edges, g):
                return n, edges
                
    # If not found, fallback to LUW
    return _luw_construct(k, g)


def _build_voltage_graph(N, M):
    """
    Build the voltage lift of K_{4,4} over Z_N with voltage matrix M.
    M[i][j] is the voltage for edge (U_i, W_j).
    """
    # Vertices:
    # U_i, a  -> index i * N + a
    # W_j, b  -> index (4 + j) * N + b
    # Total vertices: 8 * N
    
    n = 8 * N
    edges = []
    
    for i in range(4):
        for j in range(4):
            v_u = i * N # Base index for U_i
            v_w = (4 + j) * N # Base index for W_j
            voltage = M[i][j]
            
            for a in range(N):
                u_node = v_u + a
                # Edge from (U_i, a) to (W_j, b)
                # Convention: (U_i, a) -- (W_j, a + voltage)
                b = (a + voltage) % N
                w_node = v_w + b
                edges.append((u_node, w_node))
                
    return n, edges


def _has_girth_at_least(n, edges, g):
    """
    Check if the graph has girth >= g.
    Uses BFS from each vertex.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        
        for u in queue:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    if v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
                            
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) construction."""
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


# EVOLVE-BLOCK-END
