import random
import time
import sys
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # 5s margin
    
    # Target: 4-regular graph with girth 10.
    # Use voltage graph lift of a 32-vertex 4-regular bipartite graph with girth 6.
    # Voltage group: Z3 x Z3 (size 9).
    # Lifted graph will have 32 * 9 = 288 vertices.
    # Moore bound for k=4, g=10 is 243.
    # 288 is between 243 and 320.
    # If we can construct one, it's an improvement over 320.
    
    n_base = 32
    u_size = 16
    w_size = 16
    
    # Build base adjacency
    adj = [[] for _ in range(n_base)]
    for i in range(u_size):
        for j in range(4):
            w_idx = 16 + (i + j) % w_size
            adj[i].append(w_idx)
            adj[w_idx].append(i)
    
    # Enumerate all simple cycles of length 6 and 8
    def find_simple_cycles(length):
        cycles = set()
        for start in range(n_base):
            path = [start]
            visited = {start}
            
            def dfs(current, depth):
                if time.time() > deadline:
                    return
                if depth == length:
                    if current == start:
                        # Normalize cycle: rotate so smallest node is first
                        cyc = path[:]
                        min_idx = cyc.index(min(cyc))
                        normalized = tuple(cyc[min_idx:] + cyc[:min_idx])
                        cycles.add(normalized)
                    return
                
                for neighbor in adj[current]:
                    if neighbor == start and depth < length - 1:
                        continue
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, depth + 1)
                    path.pop()
                    visited.remove(neighbor)
            
            dfs(start, 1)
        return list(cycles)
    
    cycles_6 = find_simple_cycles(6)
    cycles_8 = find_simple_cycles(8)
    
    # Build edge list and index
    edge_list = []
    for i in range(n_base):
        for j in adj[i]:
            if i < j:
                edge_list.append((i, j))
                
    edge_index = {}
    for idx, (u, v) in enumerate(edge_list):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # Precompute edge contributions for each cycle
    # For each cycle, store list of (edge_idx, sign) where sign is +1 or -1
    # based on traversal direction relative to edge_list orientation
    
    def get_cycle_edges(cycle):
        """Returns list of (edge_idx, sign) for the cycle"""
        result = []
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i+1]
            e_idx = edge_index[(u, v)]
            # If u < v in edge_list, sign is +1; else -1
            eu, ev = edge_list[e_idx]
            if u == eu:
                sign = 1
            else:
                sign = -1
            result.append((e_idx, sign))
        return result
    
    cycle_edges_6 = [get_cycle_edges(c) for c in cycles_6]
    cycle_edges_8 = [get_cycle_edges(c) for c in cycles_8]
    
    # Convert cycles to constraint vectors over Z3
    # Each constraint: sum of signed voltages != 0 (mod 3)
    # We need: for each cycle, sum_{e in cycle} sign_e * voltage_e != (0,0) mod 3
    
    # Random search with better initialization
    best_voltages = None
    attempts = 0
    
    # First try: structured assignment
    # Assign voltages based on edge position
    for attempt in range(50):
        if time.time() > deadline:
            break
        # Use a structured pattern
        base_a = attempt % 3
        base_b = (attempt // 3) % 3
        voltages = []
        for idx, (u, v) in enumerate(edge_list):
            # Pattern based on node indices
            va = (u + v + base_a) % 3
            vb = (u - v + base_b) % 3
            voltages.append((va, vb))
        
        valid = True
        for ce in cycle_edges_6:
            sum_a = 0
            sum_b = 0
            for e_idx, sign in ce:
                va, vb = voltages[e_idx]
                if sign == 1:
                    sum_a = (sum_a + va) % 3
                    sum_b = (sum_b + vb) % 3
                else:
                    sum_a = (sum_a - va) % 3
                    sum_b = (sum_b - vb) % 3
            if sum_a == 0 and sum_b == 0:
                valid = False
                break
        
        if valid:
            for ce in cycle_edges_8:
                sum_a = 0
                sum_b = 0
                for e_idx, sign in ce:
                    va, vb = voltages[e_idx]
                    if sign == 1:
                        sum_a = (sum_a + va) % 3
                        sum_b = (sum_b + vb) % 3
                    else:
                        sum_a = (sum_a - va) % 3
                        sum_b = (sum_b - vb) % 3
                if sum_a == 0 and sum_b == 0:
                    valid = False
                    break
        
        if valid:
            best_voltages = voltages
            break
    
    # If structured didn't work, try random with local search
    if best_voltages is None:
        while time.time() < deadline:
            attempts += 1
            voltages = [(random.randint(0, 2), random.randint(0, 2)) for _ in range(len(edge_list))]
            
            # Check and count violations
            violations = []
            for ce in cycle_edges_6:
                sum_a = 0
                sum_b = 0
                for e_idx, sign in ce:
                    va, vb = voltages[e_idx]
                    if sign == 1:
                        sum_a = (sum_a + va) % 3
                        sum_b = (sum_b + vb) % 3
                    else:
                        sum_a = (sum_a - va) % 3
                        sum_b = (sum_b - vb) % 3
                if sum_a == 0 and sum_b == 0:
                    violations.append(('6', ce, sum_a, sum_b))
            
            for ce in cycle_edges_8:
                sum_a = 0
                sum_b = 0
                for e_idx, sign in ce:
                    va, vb = voltages[e_idx]
                    if sign == 1:
                        sum_a = (sum_a + va) % 3
                        sum_b = (sum_b + vb) % 3
                    else:
                        sum_a = (sum_a - va) % 3
                        sum_b = (sum_b - vb) % 3
                if sum_a == 0 and sum_b == 0:
                    violations.append(('8', ce, sum_a, sum_b))
            
            if not violations:
                best_voltages = voltages
                break
            
            # Local search: try to fix violations
            if attempts % 10 == 0 and violations:
                # Pick a random violation and modify an edge in that cycle
                viol = random.choice(violations)
                ce = viol[1]
                e_idx, sign = random.choice(ce)
                old_va, old_vb = voltages[e_idx]
                new_va = (old_va + 1) % 3
                new_vb = (old_vb + 1) % 3
                voltages[e_idx] = (new_va, new_vb)
    
    if best_voltages is None:
        return _luw_construct(k, g)
        
    # Construct lifted graph
    n_lifted = n_base * 9
    edges = []
    
    def node_index(b, va, vb):
        return b * 9 + va * 3 + vb
        
    for b in range(n_base):
        for va in range(3):
            for vb in range(3):
                idx = node_index(b, va, vb)
                for neighbor in adj[b]:
                    e_idx = edge_index[(b, neighbor)]
                    eva, evb = best_voltages[e_idx]
                    if b < neighbor:
                        nva = (va + eva) % 3
                        nvb = (vb + evb) % 3
                    else:
                        nva = (va - eva) % 3
                        nvb = (vb - evb) % 3
                    n_idx = node_index(neighbor, nva, nvb)
                    if idx < n_idx:
                        edges.append((idx, n_idx))
                        
    return n_lifted, edges

def _luw_construct(k, g):
    # Original LUW construction as fallback
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
