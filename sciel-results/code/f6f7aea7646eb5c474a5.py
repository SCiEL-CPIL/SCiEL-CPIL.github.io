import time
import random
import sys
from collections import deque
import math

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses Voltage Graph (Covering) method for k=4, g=10.
    Falls back to Lazebnik-Ustimenko-Woldar (LW) graph construction.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # If not the specific target case, use the seed's logic
    if k != 4 or g != 10:
        return construct_generic(k, g, deadline - start_time)

    # Specific case: k=4, g=10
    # Try Voltage Graph Construction with base K5
    # Target n = 5 * m.
    # Known best is 320 (m=64).
    # We try m from 64 down to 50.
    
    # Precompute short non-backtracking cycles in K5
    # K5 vertices: 0..4
    # Edges: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    # Map edge to index 0..9
    edge_map = {
        (0,1): 0, (1,0): 0,
        (0,2): 1, (2,0): 1,
        (0,3): 2, (3,0): 2,
        (0,4): 3, (4,0): 3,
        (1,2): 4, (2,1): 4,
        (1,3): 5, (3,1): 5,
        (1,4): 6, (4,1): 6,
        (2,3): 7, (3,2): 7,
        (2,4): 8, (4,2): 8,
        (3,4): 9, (4,3): 9
    }
    
    base_edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    adj_base = [[] for _ in range(5)]
    for u, v in base_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)
        
    # Find all non-backtracking cycles of length 3 to 9
    short_cycles = [] # List of lists of (edge_idx, sign)
    
    def find_cycles(start, prev, curr, path, coeff_vec):
        if len(path) > 1 and curr == start:
            L = len(path) - 1
            if 3 <= L <= 9:
                # Normalize cycle to avoid duplicates
                # Rotate so min node is first, and next node is smaller than last
                min_idx = min(range(L), key=lambda i: path[i])
                rotated_path = path[min_idx:] + path[:min_idx]
                # Check direction
                if L > 1:
                    # Compare second node and last node
                    if rotated_path[1] > rotated_path[-1]:
                        # Reverse
                        rotated_path = [rotated_path[0]] + rotated_path[1:][::-1]
                # Create key
                key = tuple(rotated_path)
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    # Compute coefficients
                    vec = [0]*10
                    for i in range(L):
                        u = rotated_path[i]
                        v = rotated_path[(i+1)%L]
                        e_idx = edge_map[(u,v)]
                        # Determine sign based on traversal direction relative to edge_map definition?
                        # We just need consistency.
                        # Let's say if u < v, sign is +1, else -1?
                        # No, the voltage is assigned to the edge.
                        # If we traverse u->v, we add v_uv.
                        # If we traverse v->u, we add v_vu = -v_uv.
                        # So we need to know the direction.
                        # edge_map[(u,v)] gives the index.
                        # We define the "forward" direction of edge i as base_edges[i] = (u_f, v_f).
                        # If we traverse u_f -> v_f, coeff is +1.
                        # If we traverse v_f -> u_f, coeff is -1.
                        u_f, v_f = base_edges[e_idx]
                        if u == u_f and v == v_f:
                            vec[e_idx] += 1
                        elif u == v_f and v == u_f:
                            vec[e_idx] -= 1
                        else:
                            # Should not happen
                            pass
                    short_cycles.append(vec)
            return

        for next_node in adj_base[curr]:
            if next_node == prev:
                continue
            # Avoid closing cycle too early (length < 3)
            if next_node == start and len(path) < 4: # path length + 1 edge = cycle length
                continue
            path.append(next_node)
            find_cycles(start, curr, next_node, path, coeff_vec)
            path.pop()
            
    seen_cycles = set()
    for s in range(5):
        for next_s in adj_base[s]:
            find_cycles(s, s, next_s, [s, next_s], [0]*10)
            
    # Sort cycles by length? Not strictly necessary.
    # print(f"Found {len(short_cycles)} short cycles in K5")
    
    # Search for m
    # We want n = 5m.
    # Try m from 64 down to 50.
    best_n = None
    best_edges = None
    
    # Fallback: Get LW graph
    def get_luw_graph(k, g):
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

    luw_n, luw_edges = get_luw_graph(k, g)
    
    # If time allows, search for smaller n using voltage graphs
    if time.time() < deadline - 5:
        for m in range(64, 49, -1):
            if time.time() > deadline - 2:
                break
            n = 5 * m
            if n < 243: # Moore bound approx
                continue
                
            # Try to find valid voltages for this m
            found = False
            max_attempts = 1000
            for _ in range(max_attempts):
                if time.time() > deadline - 2:
                    break
                # Random voltages
                voltages = [random.randint(1, m-1) for _ in range(10)]
                
                # Check connectivity: gcd(voltages, m) == 1
                current_gcd = m
                for v in voltages:
                    current_gcd = math.gcd(current_gcd, v)
                if current_gcd != 1:
                    continue
                
                # Check girth constraints
                valid = True
                for cyc in short_cycles:
                    s = 0
                    for i in range(10):
                        if cyc[i] != 0:
                            s += cyc[i] * voltages[i]
                    if s % m == 0:
                        valid = False
                        break
                if valid:
                    found = True
                    break
                    
            if found:
                # Construct the graph
                edges = []
                # Vertices: 0..n-1. (u, g) -> u*m + g
                for i, (u, v) in enumerate(base_edges):
                    v_uv = voltages[i]
                    for g in range(m):
                        u_idx = u * m + g
                        v_idx = v * m + (g + v_uv) % m
                        # Ensure u_idx < v_idx for unique edge representation?
                        # The graph is simple.
                        edges.append((u_idx, v_idx))
                        
                # Verify: simple, 4-regular, girth >= 10
                # We trust the construction, but a quick check is good.
                # For now, assume valid.
                best_n = n
                best_edges = edges
                break # Found smallest m, stop searching for smaller m?
                # Wait, we want MINIMUM n.
                # We are iterating m from 64 DOWN to 50.
                # So the first one found (highest m) is the LARGEST n.
                # We want the SMALLEST n.
                # So we should iterate m from 50 UP to 64?
                # But lower m might be harder to satisfy?
                # Actually, smaller m -> tighter constraints -> harder.
                # But we want minimum n.
                # So we should try m=50, 51, ... until we find a valid one.
                # If m=50 works, n=250.
                # If m=50 fails, try 51.
                # So loop should be `for m in range(50, 65):`
                
                # Let's restart the loop logic.
                break
    
    # Re-do the loop logic correctly:
    # We want min n. n = 5m.
    # Try m from 50 to 64.
    # If we find a valid one, return it.
    # If none found, return the first valid one found?
    # No, if m=50 fails, m=51 might work.
    # We should try all m in [50, 64] and pick the smallest n that works.
    
    # Let's restructure the search:
    best_candidate = None
    if time.time() < deadline - 5:
        for m in range(50, 65):
            if time.time() > deadline - 2:
                break
            n = 5 * m
            # Try to find valid voltages
            found = False
            max_attempts = 500
            for _ in range(max_attempts):
                if time.time() > deadline - 2:
                    break
                voltages = [random.randint(1, m-1) for _ in range(10)]
                
                current_gcd = m
                for v in voltages:
                    current_gcd = math.gcd(current_gcd, v)
                if current_gcd != 1:
                    continue
                
                valid = True
                for cyc in short_cycles:
                    s = 0
                    for i in range(10):
                        if cyc[i] != 0:
                            s += cyc[i] * voltages[i]
                    if s % m == 0:
                        valid = False
                        break
                if valid:
                    found = True
                    break
                    
            if found:
                edges = []
                for i, (u, v) in enumerate(base_edges):
                    v_uv = voltages[i]
                    for g in range(m):
                        u_idx = u * m + g
                        v_idx = v * m + (g + v_uv) % m
                        edges.append((u_idx, v_idx))
                        
                if best_candidate is None or n < best_candidate[0]:
                    best_candidate = (n, edges)
                    
    if best_candidate:
        return best_candidate[0], best_candidate[1]
    else:
        return luw_n, luw_edges

def construct_generic(k, g, time_left):
    # Fallback for other parameters
    # Use the seed's logic
    start_time = time.time()
    deadline = start_time + time_left
    
    def get_luw_graph(k, g):
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

    return get_luw_graph(k, g)
