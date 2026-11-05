import numpy as np
import random
import time

def construct(k, g):
    # Target: 4-regular, girth 10
    # k=4, g=10
    if k != 4 or g != 10:
        # Fallback to original LUW for other parameters
        return _luw_construct(k, g)

    # Moore bound for (4,10) is 242
    # Best known is 320
    # We search from 242 upwards
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # We will try N values
    # It is known that 4-regular graphs of girth 10 exist for N=320.
    # We start at 242 to see if we can beat 320.
    
    for N in range(242, 350):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a graph of size N
        edges = _find_cage(N, k, g, max_iters=5000)
        if edges is not None:
            return N, edges
            
    # If we couldn't find one in the search, return a known valid graph
    # The LUW construction is valid but large. 
    # We should probably just return the LUW result to be safe, 
    # or a hardcoded 320-vertex graph if we had one.
    # Since we don't have a hardcoded one, and the search might fail,
    # we fall back to the original LUW code to ensure validity.
    return _luw_construct(k, g)

def _find_cage(N, k, g, max_iters=10000):
    # Generate a random k-regular graph
    A = _random_regular_graph(N, k)
    if A is None:
        return None
        
    # Rewiring loop
    for i in range(max_iters):
        # Check girth periodically
        if i % 100 == 0:
            if _has_girth_ge(A, g):
                return _adj_to_edges(A)
                
        # Perform a rewiring
        A = _rewire(A, N, k)
        
    # Final check
    if _has_girth_ge(A, g):
        return _adj_to_edges(A)
        
    return None

def _random_regular_graph(N, k):
    # Configuration model
    # Create list of stubs
    stubs = []
    for i in range(N):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    A = np.zeros((N, N), dtype=np.int8)
    
    # Pair stubs
    for i in range(0, N * k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            # Loop, invalid for simple graph, restart or swap
            # Simplest: restart
            return None
        if A[u, v] == 1:
            # Multi-edge, invalid
            return None
        A[u, v] = 1
        A[v, u] = 1
        
    return A

def _rewire(A, N, k):
    # Pick a random edge
    edges = np.argwhere(A > 0)
    if len(edges) == 0:
        return A
        
    # Only pick one direction (u < v) to avoid duplicates
    edges = edges[edges[:, 0] < edges[:, 1]]
    
    if len(edges) < 2:
        return A
        
    idx1 = random.randint(0, len(edges) - 1)
    u1, v1 = edges[idx1]
    
    # Pick another edge
    while True:
        idx2 = random.randint(0, len(edges) - 1)
        u2, v2 = edges[idx2]
        if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
            continue
        # Ensure edges are distinct
        break
        
    # Try swap (u1, v1) and (u2, v2) -> (u1, u2) and (v1, v2)
    # Or (u1, v2) and (v1, u2)
    
    # Check validity
    if A[u1, u2] == 0 and A[v1, v2] == 0 and u1 != u2 and v1 != v2:
        A[u1, v1] = 0
        A[v1, u1] = 0
        A[u2, v2] = 0
        A[v2, u2] = 0
        
        A[u1, u2] = 1
        A[u2, u1] = 1
        A[v1, v2] = 1
        A[v2, v1] = 1
        return A
        
    # Try other swap
    if A[u1, v2] == 0 and A[v1, u2] == 0 and u1 != v2 and v1 != u2:
        A[u1, v1] = 0
        A[v1, u1] = 0
        A[u2, v2] = 0
        A[v2, u2] = 0
        
        A[u1, v2] = 1
        A[v2, u1] = 1
        A[v1, u2] = 1
        A[u2, v1] = 1
        return A
        
    return A

def _has_girth_ge(A, g):
    # Check for cycles of length 3 to g-1
    # A^k diagonal > 0 implies cycle of length <= k
    # We need no cycles of length < g
    # So check k = 3 to g-1
    A_k = A.copy()
    for k in range(3, g):
        A_k = A_k @ A
        if np.any(np.diag(A_k) > 0):
            return False
    return True

def _adj_to_edges(A):
    edges = []
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append((i, j))
    return edges

# Fallback LUW construction (copied from original)
def _luw_construct(k, g):
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
