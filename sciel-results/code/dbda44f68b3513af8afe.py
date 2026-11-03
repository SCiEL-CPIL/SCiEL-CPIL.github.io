import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = float('inf')
    best_edges = None
    
    # Deterministic LUW Construction with multiple rule variants
    min_d = max(3, g // 2 - 2)
    max_d = g + 5  # Safety cap
    
    # Generate multiple rule variants to try
    rule_variants = _generate_rule_variants(k, max_d)
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 15:
            break
        
        for variant_idx, rule in enumerate(rule_variants):
            if time.time() > deadline - 15:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
            except Exception:
                continue
            
            if n_luw < best_n:
                # Verify girth using faster method
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    best_n = n_luw
                    best_edges = edges_luw
                    break  # Found valid graph for this d, try next d

    if best_n == float('inf'):
        # Fallback: Return a valid but potentially large graph
        for variant_idx, rule in enumerate(rule_variants):
            if time.time() > deadline - 5:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, max_d, rule)
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    return n_luw, edges_luw
            except Exception:
                continue
        # Last resort
        n_luw, edges_luw = _construct_luw_with_rule(k, max_d + 3, rule_variants[0])
        return n_luw, edges_luw

    return best_n, best_edges

def _get_petersen_edges():
    edges = []
    for i in range(5):
        edges.append((i, (i+1) % 5))
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i+1) % 5]))
    for i in range(5):
        edges.append((i, i+5))
    return edges

def _random_lift(base_edges, M):
    n_base = 10
    n_lift = n_base * M
    perm_map = {}
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        if key not in perm_map:
            p = list(range(M))
            random.shuffle(p)
            perm_map[key] = p
    lift_edges = []
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        p = perm_map[key]
        for j in range(M):
            u_node = u * M + j
            v_node = v * M + p[j]
            lift_edges.append((u_node, v_node))
    return lift_edges

def _has_girth_at_least_fast(n, edges, g):
    """
    Fast girth verification using non-backtracking walks from each vertex.
    Uses BFS with parent tracking to detect cycles shorter than g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do a limited BFS to find shortest cycle through it
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            # Only continue if we haven't reached the depth limit
            if dist[u] + 1 >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a potential cycle - check if it's not the tree edge
                    if u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _generate_rule_variants(k, max_d):
    """
    Generate multiple rule variants for LUW construction.
    Returns a list of rule dictionaries.
    """
    if k != 3:
        # For k != 3, use the original rule
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, max_d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return [rule]
    
    # For k=3, generate multiple variants
    variants = []
    
    # Variant 0: Original
    rule0 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule0[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule0)
    
    # Variant 1: Different starting pattern
    rule1 = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule1[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule1)
    
    # Variant 2: Shifted pattern
    rule2 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # Use different selection
        rule2[c] = [(1, c0 - 2), (c0, 1), (c0 - 1, 1), (1, c0 + 1)][r]
    variants.append(rule2)
    
    # Variant 3: Another shift
    rule3 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule3[c] = [(1, c0 - 2), (c0 - 1, 1), (1, c0 + 1), (c0, 1)][r]
    variants.append(rule3)
    
    return variants

def _construct_luw_with_rule(k, d, rule):
    """
    Construct LUW graph with a specific rule.
    """
    q = k
    if k == 3:
        q = 3
    
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

# Keep old functions for compatibility but they're no longer called from construct
def _construct_luw_optimized(k, d):
    rule = _generate_rule_variants(k, d)[0]
    return _construct_luw_with_rule(k, d, rule)

def _construct_luw_generic(k, d):
    return _construct_luw_optimized(k, d)
# EVOLVE-BLOCK-END
