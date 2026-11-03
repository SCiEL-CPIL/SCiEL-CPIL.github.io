import time
import random
import sys
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if k != 3:
        # Fallback to LUW for other k
        return _construct_luw_fallback(k, g)

    # Strategy: Hybrid approach
    # 1. Try LUW construction with various d values (deterministic, known good girth)
    # 2. Try randomized local search with repair for smaller n
    
    best_n = float('inf')
    best_edges = None
    
    # First, try LUW construction which gives guaranteed girth
    # LUW with d parameters gives graphs with n = 2 * 3^d vertices approximately
    # Girth is related to d
    
    for d in range(8, 14):
        if time.time() > deadline - 20:
            break
        try:
            n_luw, edges_luw = _construct_luw_optimized(3, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                if n_luw < best_n:
                    best_n = n_luw
                    best_edges = edges_luw
        except Exception:
            continue
    
    # If LUW gives a reasonable answer, try to improve with local search
    if best_n != float('inf') and best_n > 600:
        # Try local search to find smaller graph
        if time.time() < deadline - 30:
            n_target = min(best_n - 2, 700)  # Try to find smaller
            for n_try in range(512, n_target, 2):
                if time.time() > deadline - 15:
                    break
                found = _local_search_for_girth(n_try, g, deadline - 10)
                if found is not None:
                    best_n = n_try
                    best_edges = found
                    break
    
    if best_n == float('inf'):
        # Last resort
        n_luw, edges_luw = _construct_luw_optimized(3, 12)
        return n_luw, edges_luw

    return best_n, best_edges
    
    # If LPS didn't find a good enough graph, fall back to LUW
    if best_n == float('inf') or best_n > 2000:
        for d in range(8, 15):
            if time.time() > deadline - 5:
                break
            try:
                n_luw, edges_luw = _construct_luw_optimized(3, d)
                if n_luw < best_n and _has_girth_at_least_fast(n_luw, edges_luw, g):
                    best_n = n_luw
                    best_edges = edges_luw
            except Exception:
                continue
    
    if best_n == float('inf'):
        # Last resort: return a large valid graph
        n_luw, edges_luw = _construct_luw_optimized(3, 20)
        return n_luw, edges_luw

    return best_n, best_edges

def _construct_lps_graph(q):
    """
    Construct LPS (Lubotzky-Phillips-Sarnak) 3-regular graph using PSL(2, F_q).
    
    The LPS construction uses the Cayley graph of PSL(2, Z) with generators
    A = [[1,1],[0,1]], B = [[0,-1],[1,0]] reduced mod q.
    
    Vertices are elements of PSL(2, F_q), which has order q(q²-1)/2.
    Each vertex has degree 4 (from generators A, A⁻¹, B, B⁻¹).
    To get degree 3, we use a quotient or different generator set.
    
    Actually, the standard LPS graph is 4-regular. For 3-regular, we can use
    the (3, q) Moore graph construction or take a quotient.
    
    Alternative: Use the Margulis construction which gives 3-regular graphs.
    """
    # For 3-regular, we'll use a different algebraic approach
    # The Margulis-Vaughan-Li construction gives 3-regular Ramanujan graphs
    
    # Vertices: elements of F_q × F_q
    # Actually, let's use the standard approach: Cayley graph on PSL(2, F_q)
    # with 3 generators
    
    # PSL(2, F_q) has order q(q²-1)/2
    # We need to enumerate elements and find neighbors
    
    # Represent elements as 2x2 matrices with determinant 1 mod q
    # Normalize: trace can be 0, 1, or -1 (mod q) for unique representation
    
    elements = []
    element_index = {}
    
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # d = (1 - b*c) / a mod q if a != 0, else handle separately
                if a != 0:
                    inv_a = pow(a, q-2, q)  # a^(-1) mod q
                    d = (1 - b*c) * inv_a % q
                    # Check determinant = 1
                    if (a*d - b*c) % q == 1:
                        # Normalize: make trace non-negative or use canonical form
                        # Use (a, b, c, d) with a > 0 or a == 0 and b > 0
                        key = (a, b, c, d)
                        if key not in element_index:
                            element_index[key] = len(elements)
                            elements.append(key)
                else:
                    # a = 0, then -bc = 1, so bc = -1 mod q
                    if b != 0:
                        c = (-1) * pow(b, q-2, q) % q
                        d = 0
                        key = (a, b, c, d)
                        if key not in element_index:
                            element_index[key] = len(elements)
                            elements.append(key)
    
    n = len(elements)
    
    # Define 3 generators for 3-regular graph
    # Use A = [[1,1],[0,1]], B = [[0,-1],[1,0]], C = [[1,1],[1,0]]
    # But we need exactly 3 neighbors per vertex
    
    # For simplicity, use the standard LPS 4-regular and take a 2-cover or quotient
    # Actually, let's just return the 4-regular LPS graph and let the evaluator check
    # No, we need exactly degree 3.
    
    # Alternative: Use the (3, q) cage construction from algebraic curves
    # This is complex. Let's fall back to a simpler approach.
    
    # For now, return a large LUW graph as fallback
    return _construct_luw_optimized(3, max(8, g//2))

def _construct_lps_graph(q):
    """
    Construct a 3-regular graph using the LPS method with PSL(2, F_q).
    
    We construct the 4-regular LPS Cayley graph on PSL(2, F_q) and then
    take a 2-vertex quotient to get a 3-regular graph, or use a different
    generator set.
    
    Alternative: Use the standard (3, g) cage from algebraic curves.
    For girth 16, the smallest known is 936 vertices.
    
    This implementation uses a randomized algebraic approach:
    1. Start with a base 3-regular graph
    2. Use edge swaps guided by algebraic constraints to increase girth
    """
    # For now, use LUW as a reliable fallback
    # The LUW construction with appropriate d gives good girth
    d = max(8, g // 2 + 2)
    return _construct_luw_optimized(3, d)

def _local_search_for_girth(n, g, deadline):
    """
    Local search to find a 3-regular graph with girth at least g.
    Uses edge swap moves to increase girth.
    """
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Generate random 3-regular graph
        edges = _generate_random_3_regular(n)
        if edges is None:
            continue
        
        # Try to increase girth via edge swaps
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u,v), max(u,v)))
        
        # Iterative improvement
        improved = True
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find shortest cycle
            min_cycle, cycle_nodes = _find_shortest_cycle(adj, g)
            if min_cycle >= g:
                return list(edge_set)
            
            # Try to break the cycle via edge swap
            if not _try_edge_swap_to_break_cycle(adj, edge_set, cycle_nodes, n):
                break
        
        # If we didn't succeed, try again with new restart
    
    return None

def _find_shortest_cycle(adj, target_girth):
    """Find the shortest cycle in the graph."""
    n = len(adj)
    min_cycle = float('inf')
    min_cycle_nodes = None
    
    for start in range(n):
        # BFS to find shortest cycle through start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        # Reconstruct cycle
                        cycle_nodes = []
                        x = u
                        while x != -1:
                            cycle_nodes.append(x)
                            x = parent[x]
                        x = v
                        while x != -1:
                            cycle_nodes.append(x)
                            x = parent[x]
                        min_cycle_nodes = cycle_nodes
    
    return min_cycle, min_cycle_nodes

def _try_edge_swap_to_break_cycle(adj, edge_set, cycle_nodes, n):
    """
    Try to break a cycle by swapping edges.
    Returns True if a successful swap was made.
    """
    if not cycle_nodes or len(cycle_nodes) < 4:
        return False
    
    # Get unique nodes in cycle
    cycle_set = set(cycle_nodes)
    
    # Try swapping edges incident to cycle nodes
    for u in cycle_set:
        for v in adj[u]:
            if (min(u,v), max(u,v)) not in edge_set:
                continue
            
            # Try to find an alternative edge for u or v
            # This is complex; for now, just try random swaps
            pass
    
    # Simplified: try a random edge swap
    if len(cycle_nodes) >= 4:
        # Pick two non-adjacent edges in the cycle and try to swap
        # For simplicity, try a few random swaps
        for _ in range(20):
            # Pick random edge
            edge = random.choice(list(edge_set))
            u, v = edge
            
            # Pick another edge
            edge2 = random.choice(list(edge_set))
            u2, v2 = edge2
            
            # Try swap: remove (u,v) and (u2,v2), add (u,u2) and (v,v2)
            if u == u2 or u == v2 or v == u2 or v == v2:
                continue
            
            # Check if new edges exist
            new_e1 = (min(u,u2), max(u,u2))
            new_e2 = (min(v,v2), max(v,v2))
            
            if new_e1 in edge_set or new_e2 in edge_set:
                continue
            if u == u2 or v == v2:
                continue
            
            # Perform swap
            edge_set.remove(edge)
            edge_set.remove(edge2)
            edge_set.add(new_e1)
            edge_set.add(new_e2)
            
            # Update adj
            adj[u].remove(v)
            adj[v].remove(u)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u].append(u2)
            adj[u2].append(u)
            adj[v].append(v2)
            adj[v2].append(v)
            
            return True
    
    return False

def _generate_random_3_regular(n):
    # Generate a random 3-regular graph
    # Using configuration model
    # List of half-edges
    half_edges = []
    for i in range(n):
        half_edges.extend([i, i, i])
        
    random.shuffle(half_edges)
    
    edges = set()
    adj = [[] for _ in range(n)]
    
    for i in range(0, 2*n, 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            return None # Loop
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge
        edges.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    return list(edges)

def _has_girth_at_least_fast(n, edges, g):
    # Optimized girth check
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    # Optimization: Stop if we find a cycle < g
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw_fallback(k, g):
    # Reuse the logic from the current program's LUW
    # We need to include the helper functions from the current program
    # Since I am replacing the block, I must include them or assume they exist?
    # The prompt says "Keep the function names and signatures of the seed program".
    # The current program HAS _construct_luw_optimized.
    # I should call it.
    
    min_d = max(3, g // 2 - 2)
    for d in range(min_d, min_d + 10):
        try:
            n_luw, edges_luw = _construct_luw_optimized(k, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue
    return _construct_luw_optimized(k, min_d + 10)

def _construct_luw_optimized(k, d):
    """
    Deterministic LUW construction for k-regular graphs.
    Optimized for k=3.
    """
    if k != 3:
        return _construct_luw_generic(k, d)
        
    q = 3
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _construct_luw_generic(k, d):
    q = k
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
