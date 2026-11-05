import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)

    start_time = time.time()
    time_limit = 110.0
    
    # The best known graph has 320 vertices.
    # We focus our search here. Searching for N < 320 is likely futile and wastes time.
    target_N = 320
    
    # Try to find a valid graph of size 320
    edges = _find_cage_fast(target_N, k, g, max_iters=50000)
    if edges is not None:
        return target_N, edges

    # If we fail to find a 320-vertex graph (which is unlikely if the algorithm is good, 
    # but possible due to randomness), we fall back to the LUW construction.
    # The LUW construction produces a valid graph, albeit larger (N ~ 2048).
    return _luw_construct(k, g)

def _find_cage_fast(N, k, g, max_iters=50000):
    # Generate a random k-regular graph
    A = _random_regular_graph(N, k)
    if A is None:
        return None
        
    # Precompute adjacency lists for faster BFS
    adj = _build_adj_lists(A, N)
    
    for i in range(max_iters):
        # Check girth using BFS (much faster than matrix multiplication)
        min_cycle_len = _min_girth_bfs(adj, N)
        
        if min_cycle_len >= g:
            return _adj_to_edges(A)
            
        # If we have a short cycle, we need to break it.
        # Simple strategy: Random rewiring.
        # To be smarter, we could target edges in the short cycle, 
        # but random rewiring often works if we do enough iterations.
        A, adj = _rewire_fast(A, adj, N, k)
        
        if time.time() - start_time > 110.0:
            break

    return None

def _build_adj_lists(A, N):
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j]:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def _min_girth_bfs(adj, N):
    # Returns the length of the shortest cycle
    # If no cycle, returns infinity (or a large number)
    min_len = float('inf')
    
    for start in range(N):
        # BFS from start
        # dist[v] = distance from start
        # parent[v] = previous node in BFS tree
        dist = [-1] * N
        parent = [-1] * N
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # If v is already visited and v is not the parent of u
                    # and u is not the parent of v (to avoid counting the edge twice)
                    if v != parent[u] and u != parent[v]:
                        # We found a cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_len:
                            min_len = cycle_len
                            
    return min_len

def _rewire_fast(A, adj, N, k):
    # Pick two random edges and swap them
    # We need to maintain the adjacency list consistency.
    # For simplicity and speed, we will rebuild the adjacency list if a rewiring happens.
    # Or we can update it incrementally.
    
    # Get a list of edges
    edges = []
    for i in range(N):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
                
    if len(edges) < 2:
        return A, adj
        
    idx1 = random.randint(0, len(edges) - 1)
    u1, v1 = edges[idx1]
    
    while True:
        idx2 = random.randint(0, len(edges) - 1)
        u2, v2 = edges[idx2]
        if (u1, v1) != (u2, v2) and (u1, v1) != (v2, u2):
            break
            
    # Attempt swap 1: (u1, v1) and (u2, v2) -> (u1, u2) and (v1, v2)
    if A[u1, u2] == 0 and A[v1, v2] == 0 and u1 != u2 and v1 != v2:
        # Remove old edges
        A[u1, v1] = 0; A[v1, u1] = 0
        A[u2, v2] = 0; A[v2, u2] = 0
        
        # Add new edges
        A[u1, u2] = 1; A[u2, u1] = 1
        A[v1, v2] = 1; A[v2, v1] = 1
        
        # Update adjacency lists
        adj[u1].remove(v1); adj[v1].remove(u1)
        adj[u2].remove(v2); adj[v2].remove(u2)
        
        adj[u1].append(u2); adj[u2].append(u1)
        adj[v1].append(v2); adj[v2].append(v1)
        
        return A, adj
        
    # Attempt swap 2: (u1, v1) and (u2, v2) -> (u1, v2) and (v1, u2)
    if A[u1, v2] == 0 and A[v1, u2] == 0 and u1 != v2 and v1 != u2:
        # Remove old edges
        A[u1, v1] = 0; A[v1, u1] = 0
        A[u2, v2] = 0; A[v2, u2] = 0
        
        # Add new edges
        A[u1, v2] = 1; A[v2, u1] = 1
        A[v1, u2] = 1; A[u2, v1] = 1
        
        # Update adjacency lists
        adj[u1].remove(v1); adj[v1].remove(u1)
        adj[u2].remove(v2); adj[v2].remove(u2)
        
        adj[u1].append(v2); adj[v2].append(u1)
        adj[v1].append(u2); adj[u2].append(v1)
        
        return A, adj
        
    return A, adj

def _random_regular_graph(N, k):
    stubs = []
    for i in range(N):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    A = np.zeros((N, N), dtype=np.int8)
    
    for i in range(0, N * k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if A[u, v] == 1:
            return None
        A[u, v] = 1
        A[v, u] = 1
        
    return A

def _adj_to_edges(A):
    edges = []
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append((i, j))
    return edges

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
