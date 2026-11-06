"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict
import random

# EVOLVE-BLOCK-START

def _has_c4(n, adj):
    """Check if graph has a C4 using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # More than one bit set
                return True
    return False

def _max_independent_set_size(n, adj, target=11):
    """Check if there exists an independent set of size >= target using branching with pruning."""
    def search(vertices, current_size, best):
        if current_size + len(vertices) <= best:
            return best
        if not vertices:
            return max(best, current_size)
        if current_size >= target:
            return target
        
        # Upper bound: max possible independent set from remaining
        # Use a simple bound: count of non-adjacent vertices
        ub = 0
        for i, v in enumerate(vertices):
            # Count vertices in vertices that are not adjacent to v
            non_adj = sum(1 for u in vertices if u != v and not ((adj[v] >> u) & 1))
            ub = max(ub, non_adj)
        if current_size + ub <= best:
            return best
        
        # Pick vertex with highest degree in the subgraph
        v = vertices[0]
        rest = vertices[1:]
        
        # Option 1: include v
        new_rest = [u for u in rest if not ((adj[v] >> u) & 1)]
        best = search(new_rest, current_size + 1, best)
        
        # Option 2: exclude v
        best = search(rest, current_size, best)
        
        return best
    
    return search(list(range(n)), 0, 0)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4(n, adj):
        return False
    
    if _max_independent_set_size(n, adj, 11) >= 11:
        return False
    
    return True

def _build_polarity_graph(q):
    """
    Build the polarity graph of a projective plane of order q.
    Vertices are (x, y) for x, y in range(q) plus special vertices.
    This gives q^2 + q + 1 vertices, C4-free.
    """
    n = q * q + q + 1
    adj = [0] * n
    edges = []
    
    # Map (i, j) to vertex index: i * q + j for i, j in range(q)
    # Special vertices: q^2 + i for i in range(q+1)
    
    def idx(i, j):
        return i * q + j
    
    def sidx(i):
        return q * q + i
    
    # In a polarity graph of PG(2, q), two points are adjacent if they are orthogonal
    # For the standard polarity, point (x, y, 1) is orthogonal to (x', y', 1) iff x*x' + y*y' + 1 = 0
    # This requires q to be such that -1 is a square, or we use a different polarity
    
    # For simplicity, use a known C4-free construction:
    # The graph where vertices are elements of Z_q x Z_q, and (a,b) ~ (c,d) iff (a-c)^2 + (b-d)^2 = 1 in F_q
    # This is C4-free for prime q.
    
    if q == 5:
        # Z_5 x Z_5, edges when squared distance = 1
        for i in range(q):
            for j in range(q):
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        if (di * di + dj * dj) % q == 1:
                            if di == 0 and dj == 0:
                                continue
                            ni = (i + di) % q
                            nj = (j + dj) % q
                            if ni > i or (ni == i and nj > j):
                                u = idx(i, j)
                                v = idx(ni, nj)
                                if not ((adj[u] >> v) & 1):
                                    edges.append([u, v])
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
        
        # Add q+1 special vertices connected to form the full polarity graph
        # For now, just return the q^2 part which is already C4-free
        return q * q, edges, adj
    
    return 0, [], []

def _greedy_construct(n_target):
    """Greedy construction of a C4-free graph."""
    adj = [0] * n_target
    edges = []
    
    all_pairs = list(itertools.combinations(range(n_target), 2))
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        common = adj[u] & adj[v]
        if common & (common - 1):
            continue
        edges.append([u, v])
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    return n_target, edges, adj

def _local_search(n, adj, edges, deadline):
    """Local search to minimize independence number."""
    best_alpha = _max_independent_set_size(n, adj, 11)
    best_adj = adj[:]
    best_edges = edges[:]
    
    while time.time() < deadline:
        improved = False
        
        for _ in range(100):
            if time.time() > deadline:
                break
            
            u, v = random.sample(range(n), 2)
            if (adj[u] >> v) & 1:
                continue
            
            common = adj[u] & adj[v]
            if common & (common - 1):
                continue
            
            new_adj = adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            new_alpha = _max_independent_set_size(n, new_adj, 11)
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_adj = new_adj
                best_edges = edges + [[u, v]]
                adj = new_adj
                improved = True
                if best_alpha < 11:
                    return best_adj, best_edges
        
        if not improved:
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                if not edges:
                    break
                
                idx = random.randint(0, len(edges) - 1)
                u, v = edges[idx]
                
                new_adj = adj[:]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
                new_edges = edges[:idx] + edges[idx+1:]
                all_pairs = list(itertools.combinations(range(n), 2))
                random.shuffle(all_pairs)
                
                for a, b in all_pairs:
                    if time.time() > deadline:
                        break
                    if (new_adj[a] >> b) & 1:
                        continue
                    common = new_adj[a] & new_adj[b]
                    if common & (common - 1):
                        continue
                    new_adj[a] |= (1 << b)
                    new_adj[b] |= (1 << a)
                    new_edges.append([a, b])
                
                new_alpha = _max_independent_set_size(n, new_adj, 11)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_adj = new_adj
                    best_edges = new_edges
                    adj = new_adj
                    edges = new_edges
                    improved = True
                    if best_alpha < 11:
                        return best_adj, best_edges
    
    return best_adj, best_edges

def _try_construction():
    """Try to find a C4-free graph with no independent set of size 11."""
    start_time = time.time()
    deadline = start_time + 110
    
    best_n = 0
    best_edges = []
    best_adj = None
    
    for n_try in [39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + n_try)
            
            n, edges, adj = _greedy_construct(n_try)
            
            alpha = _max_independent_set_size(n, adj, 11)
            
            if alpha < 11:
                if n_try > best_n:
                    best_n = n_try
                    best_edges = edges
                    best_adj = adj
                    if n_try >= 39:
                        return {'n': best_n, 'edges': best_edges}
            else:
                adj, edges = _local_search(n, adj, edges, deadline)
                alpha = _max_independent_set_size(n, adj, 11)
                if alpha < 11:
                    if n_try > best_n:
                        best_n = n_try
                        best_edges = edges
                        best_adj = adj
                        if n_try >= 39:
                            return {'n': best_n, 'edges': best_edges}
        
        if best_n >= n_try:
            break
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    
    result = _try_construction()
    
    if _verify_graph(result['n'], result['edges']):
        return result
    
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
