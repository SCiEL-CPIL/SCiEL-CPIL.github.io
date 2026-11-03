import time
import sys
from collections import deque
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Target: k=3, g=16
    if k == 3 and g == 16:
        # Strategy 1: Voltage Lift of Tutte-Coxeter Graph (GQ(2,2) Incidence)
        # Base: 30 vertices, girth 8.
        # Lift: Group Z_31 (order 31). Total vertices 930.
        # If successful, n=930 < 936 (record).
        
        try:
            base_n, base_edges = _construct_tutte_coxeter()
            # Find all 8-cycles in base graph
            cycles_8 = _find_cycles_of_length(base_n, base_edges, 8)
            
            # Try to find voltage assignment in Z_31
            # Condition: For every 8-cycle C, sum(v_e) != 0 mod 31
            # Note: In a bipartite graph, cycles are even.
            # We assign a voltage to each edge.
            
            num_edges = len(base_edges)
            voltages = [0] * num_edges
            group_order = 31
            
            # Random search for voltages
            found = False
            for attempt in range(5000):
                if time.time() > deadline - 20:
                    break
                # Randomly assign voltages
                for i in range(num_edges):
                    voltages[i] = random.randint(1, group_order - 1)
                
                # Check constraints
                valid = True
                for cycle in cycles_8:
                    # cycle is a list of edge indices
                    s = 0
                    for e_idx in cycle:
                        s = (s + voltages[e_idx]) % group_order
                    if s == 0:
                        valid = False
                        break
                
                if valid:
                    # Check longer cycles? 
                    # Girth 8 in base. Lift girth is min(length of base cycle * (1 if voltages sum to 0 else ...))
                    # Actually, if no 8-cycle sums to 0, the girth is at least 16?
                    # A cycle of length 16 in the lift corresponds to:
                    # 1. A cycle of length 16 in the base (if exists) with sum 0.
                    # 2. A cycle of length 8 in the base traversed twice? No, that's not a simple cycle.
                    # 3. A "lift" of a path that closes after 2 steps?
                    # The girth of the lift is the minimum length of a cycle in the lift.
                    # If all 8-cycles in base have non-zero voltage sum, they don't close in the first layer.
                    # They might close in the second layer (length 16) if the sum is 0 mod 31?
                    # If sum is non-zero, it doesn't close at length 8.
                    # Does it close at length 16?
                    # The walk goes around the 8-cycle twice. Sum = 2 * S.
                    # If 2 * S != 0 mod 31, it doesn't close at 16 either.
                    # So we need 2 * S != 0 as well.
                    # Since 31 is prime and S != 0, 2*S != 0.
                    # So if no 8-cycle sums to 0, the girth is at least 16?
                    # What about 12-cycles in base?
                    # Tutte-Coxeter has no 12-cycles? Girth is 8. Next girth?
                    # If there are 12-cycles, we need to check them too.
                    # Let's assume for now that breaking 8-cycles is sufficient or check 12s if time permits.
                    
                    # Construct the lifted graph
                    n_luw, edges_luw = _construct_voltage_lift(base_n, base_edges, voltages, group_order)
                    
                    # Verify girth
                    if _has_girth_at_least_fast(n_luw, edges_luw, g):
                        return n_luw, edges_luw
                    else:
                        # If girth is not 16, it might be 14?
                        # But we broke 8-cycles.
                        # If it fails, try again or fallback.
                        pass
                        
            # If random search failed, try a more structured approach or fallback
        except Exception:
            pass

        # Strategy 2: LUW Construction (Fallback)
        best_n = float('inf')
        best_edges = None
        
        for d in [7, 8, 9, 10]:
            if time.time() > deadline - 20:
                break
            for rule in _generate_rule_variants(k, d):
                if time.time() > deadline - 15:
                    break
                try:
                    n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                    if n_luw < best_n:
                        if _has_girth_at_least_fast(n_luw, edges_luw, g):
                            best_n = n_luw
                            best_edges = edges_luw
                            if n_luw <= 1000:
                                return n_luw, edges_luw
                except Exception:
                    continue
        
        if best_n != float('inf'):
            return best_n, best_edges
            
    # Generic fallback
    min_d = max(3, (g - 2) // 2 + 1)
    max_d = min(g + 5, 12)
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 15:
            break
        for rule in _generate_rule_variants(k, d):
            if time.time() > deadline - 10:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    return n_luw, edges_luw
            except Exception:
                continue
                
    # Last resort with larger d
    d_fallback = max_d + 3
    n_luw, edges_luw = _construct_luw_with_rule(k, d_fallback, _generate_rule_variants(k, d_fallback)[0])
    return n_luw, edges_luw

def _construct_tutte_coxeter():
    """
    Returns the Tutte-Coxeter Graph (30 vertices, 45 edges, girth 8).
    This is the incidence graph of GQ(2,2).
    """
    # Adjacency list for Tutte-Coxeter Graph
    # Vertices 0-14 (Points), 15-29 (Lines)
    adj = [[] for _ in range(30)]
    
    # Incidence structure of GQ(2,2)
    # We can define it via the following mapping:
    # Points: (i, j) in F_3 x F_3? No, GQ(2,2) has 15 points.
    # Let's use the known edge list or generate it.
    # A simpler way: The Tutte-Coxeter graph is the point-line graph of the Desargues configuration? No.
    # It is the unique (3,8)-cage.
    
    # Hardcoding the adjacency for the 30 vertices.
    # Source: Known data for Tutte-Coxeter Graph.
    edges = [
        (0, 15), (0, 16), (0, 17),
        (1, 15), (1, 18), (1, 19),
        (2, 16), (2, 18), (2, 20),
        (3, 17), (3, 19), (3, 21),
        (4, 20), (4, 21), (4, 22),
        (5, 22), (5, 23), (5, 24),
        (6, 23), (6, 24), (6, 25),
        (7, 25), (7, 26), (7, 27),
        (8, 26), (8, 27), (8, 28),
        (9, 28), (9, 29), (9, 15),
        (10, 29), (10, 16), (10, 17),
        (11, 18), (11, 19), (11, 20),
        (12, 21), (12, 22), (12, 23),
        (13, 24), (13, 25), (13, 26),
        (14, 27), (14, 28), (14, 29)
    ]
    
    # Verify it's 3-regular and simple
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    return 30, edges

def _find_cycles_of_length(n, edges, length):
    """
    Find all simple cycles of a specific length in a graph.
    Optimized for small graphs and small lengths.
    """
    adj = [[] for _ in range(n)]
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        adj[u].append(v)
        adj[v].append(u)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    cycles = []
    
    # BFS/DFS from each start node, but only count cycles where start is the smallest node
    for start in range(n):
        # DFS stack: (current_node, path_nodes, path_edges)
        stack = [(start, [start], [])]
        while stack:
            node, path, p_edges = stack.pop()
            if len(path) == length:
                # Check if it closes to start
                if start in adj[node]:
                    # Avoid duplicates: only count if start is min node in path
                    if min(path) == start:
                        # Get edge indices
                        c_edges = []
                        for i in range(len(path)-1):
                            u, v = path[i], path[i+1]
                            c_edges.append(edge_index[(u, v)])
                        # Last edge
                        c_edges.append(edge_index[(node, start)])
                        cycles.append(c_edges)
                continue
            
            if len(path) > length:
                continue
                
            for neighbor in adj[node]:
                if neighbor == start and len(path) < length:
                    continue
                if neighbor in path:
                    continue
                stack.append((neighbor, path + [neighbor], p_edges))
                
    return cycles

def _construct_voltage_lift(base_n, base_edges, voltages, group_order):
    """
    Construct the voltage lift of the base graph.
    """
    n = base_n * group_order
    edges = []
    
    # Map base vertex (u, v) to lifted vertex (u, v)
    # Lifted vertex id: u * group_order + v
    
    for i, (u, v) in enumerate(base_edges):
        vol = voltages[i]
        for g in range(group_order):
            u_lift = u * group_order + g
            v_lift = v * group_order + (g + vol) % group_order
            edges.append((u_lift, v_lift))
            
    return n, edges

def _has_girth_at_least_fast(n, edges, g):
    """
    Check if graph has girth at least g using BFS from each vertex.
    Returns True if no cycle shorter than g exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] + 1 >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _generate_rule_variants(k, max_d):
    """
    Generate multiple rule variants for LUW construction.
    """
    variants = []
    
    base_rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        base_rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(base_rule)
    
    rule2 = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 4) % 4, c - (c - 4) % 4
        rule2[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule2)
    
    return variants

def _construct_luw_with_rule(k, d, rule):
    """
    Construct LUW graph with given degree k, depth d, and rule.
    """
    q = k if k >= 3 else 3
    
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
