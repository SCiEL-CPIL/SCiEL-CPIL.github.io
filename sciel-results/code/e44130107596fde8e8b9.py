"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque
import numpy as np

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False
        
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check girth using non-backtracking BFS from each vertex
    # Optimized: We only need to check up to depth g-1
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if depth >= g - 1:
                continue # We only care about cycles shorter than g, so depth+1 < g => depth < g-1
                
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], 
                 [0, 1, 2, 3], 
                 [0, 2, 3, 1], 
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

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

def _count_short_cycles_np(adj_np, n, g):
    """
    Count cycles of length < g using NumPy adjacency matrix.
    Only counts 4 and 6 cycles efficiently for g=10.
    For general g, falls back to Python if g is small, but we optimize for the 4,10 case.
    """
    if g <= 4:
        return 0 # No cycles shorter than 4 in simple graph? Triangle is 3.
    
    count = 0
    
    # Count 4-cycles
    # C4 = 1/8 * tr(A^4) - 1/2 * sum(d_i * (d_i-1))
    # For sparse graphs, intersection method is faster:
    # C4 = sum_{i<j} binom(|N(i) & N(j)|, 2)
    # We can do this with numpy:
    # adj_np is 0/1 matrix.
    # Inter = adj_np @ adj_np
    # Diagonal of Inter is degree.
    # Off-diagonal Inter[i,j] is |N(i) & N(j)|.
    # C4 = 0.5 * (sum(Inter^2) - trace(Inter^2)) / 2 ? 
    # Let's use the explicit formula:
    # Number of 4-cycles = (1/8) * (tr(A^4) - 2 * sum_i d_i(d_i-1) - 4 * sum_{i<j} (A^2)_{ij})? No.
    
    # Simpler:
    # A2 = adj_np @ adj_np
    # A4 = A2 @ A2
    # tr_A4 = np.trace(A4)
    # sum_d2 = np.sum(adj_np.sum(axis=1) * (adj_np.sum(axis=1) - 1))
    # off_diag_A2 = np.sum(A2) - np.trace(A2)
    # C4 = (tr_A4 - 2*sum_d2 - 2*off_diag_A2) / 8 ? 
    # Let's verify for a single 4-cycle.
    # A^4 trace = 8 (4 cycles * 2 directions * 2 start points? No. 1 cycle of length 4 has 8 closed walks of length 4: u-v-w-x-u, u-x-w-v-u, etc.)
    # sum_d2: each vertex degree 2 in the cycle? No, degree is 4 in graph.
    # This formula is tricky.
    
    # Let's just count 4-cycles by checking pairs of neighbors.
    # For speed, we assume n is small (320).
    # We will use the Python set intersection but vectorized? No, Python sets are fast enough for 320 nodes.
    # But we are in a tight loop.
    
    # Let's stick to the Python set intersection for 4-cycles, it's O(n^2 * d).
    adj_list = [list(np.where(adj_np[i] > 0)[0]) for i in range(n)]
    adj_set = [set(l) for l in adj_list]
    
    c4 = 0
    for i in range(n):
        for j in range(i + 1, n):
            common = len(adj_set[i] & adj_set[j])
            if common >= 2:
                c4 += common * (common - 1) // 2
                
    count += c4
    
    if g <= 6:
        return count
        
    # Count 6-cycles
    # This is much harder. 
    # For the annealing, we might just rely on the 4-cycle count to get close, 
    # and then do a full validation at the end.
    # However, 6-cycles are a major obstacle.
    # We can estimate 6-cycles by sampling? No.
    # We will return c4 as the primary cost, and add a penalty if we detect a 6-cycle during a spot check?
    # No, let's just minimize 4-cycles first.
    
    return count

def _random_4_regular(n, seed):
    """
    Generate a random 4-regular simple graph on n vertices.
    """
    rng = random.Random(seed)
    
    for attempt in range(50):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        rng.shuffle(stubs)
        
        edges = set()
        valid = True
        adj_count = [0] * n
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
            adj_count[u] += 1
            adj_count[v] += 1
            if adj_count[u] > 4 or adj_count[v] > 4:
                valid = False
                break
        
        if valid and all(c == 4 for c in adj_count):
            return list(edges)
            
    return None

def _improve_girth_fast(n, edges, g, time_limit, rng):
    """
    Use simulated annealing with NumPy-accelerated cost to increase girth.
    """
    adj_np = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj_np[u, v] = 1
        adj_np[v, u] = 1
        
    current_cost = _count_short_cycles_np(adj_np, n, g)
    if current_cost == 0:
        return edges, 0
        
    best_edges = list(edges)
    best_cost = current_cost
    
    t_start = time.time()
    temp = 10.0
    cooling = 0.9995
    
    edge_list = list(edges)
    
    while time.time() < t_start + time_limit:
        # Pick two random edges
        e1 = rng.choice(edge_list)
        e2 = rng.choice(edge_list)
        if e1 == e2:
            continue
            
        u1, v1 = e1
        u2, v2 = e2
        
        # Try swap: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2)
        # Check validity
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
            
        # Check if new edges already exist
        if adj_np[u1, u2] == 1 or adj_np[v1, v2] == 1:
            continue
            
        # Apply swap
        adj_np[u1, v1] = 0
        adj_np[v1, u1] = 0
        adj_np[u2, v2] = 0
        adj_np[v2, u2] = 0
        
        adj_np[u1, u2] = 1
        adj_np[u2, u1] = 1
        adj_np[v1, v2] = 1
        adj_np[v2, v1] = 1
        
        # Update edge list
        edge_list.remove(e1)
        edge_list.remove(e2)
        new_e1 = (min(u1, u2), max(u1, u2))
        new_e2 = (min(v1, v2), max(v1, v2))
        edge_list.append(new_e1)
        edge_list.append(new_e2)
        
        new_cost = _count_short_cycles_np(adj_np, n, g)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_edges = list(edge_list)
            
        if new_cost <= current_cost or rng.random() < pow(2.718, -(new_cost - current_cost) / temp):
            current_cost = new_cost
        else:
            # Revert
            adj_np[u1, u2] = 0
            adj_np[u2, u1] = 0
            adj_np[v1, v2] = 0
            adj_np[v2, v1] = 0
            
            adj_np[u1, v1] = 1
            adj_np[v1, u1] = 1
            adj_np[u2, v2] = 1
            adj_np[v2, u2] = 1
            
            edge_list.remove(new_e1)
            edge_list.remove(new_e2)
            edge_list.append(e1)
            edge_list.append(e2)
            
        temp *= cooling
        
    return best_edges, best_cost

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # 1. Try LUW Graph as fallback
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. If k=4, g=10, try to find a smaller graph (Foster graph, n=320)
    if k == 4 and g == 10:
        target_n = 320
        rng = random.Random(42)
        
        # Try a few random starts
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            edges = _random_4_regular(target_n, seed=100 + attempt)
            if edges is None:
                continue
                
            # Check if it's already valid (unlikely)
            if _validate_graph(target_n, edges, k, g):
                best_n = target_n
                best_edges = edges
                break
                
            # Try to improve girth
            time_left = deadline - time.time()
            if time_left < 5:
                break
                
            # Allocate 80% of remaining time to annealing
            anneal_time = time_left * 0.8
            improved_edges, cost = _improve_girth_fast(target_n, edges, g, anneal_time, rng)
            
            if cost == 0:
                if _validate_graph(target_n, improved_edges, k, g):
                    best_n = target_n
                    best_edges = improved_edges
                    break
            else:
                # If we didn't get cost 0, we might still have a better graph than LUW
                # But we must ensure validity.
                # We only return 320 if it's valid.
                pass

    return best_n, best_edges

# EVOLVE-BLOCK-END
