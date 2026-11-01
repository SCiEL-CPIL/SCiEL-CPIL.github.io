import time
import random
import sys
import math
import numpy as np

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(d, q):
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

    def has_girth_at_least(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
            
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        # Try to construct the 320-vertex cage using voltage graphs
        # Base graph: K5,5 minus perfect matching (10 vertices, 4-regular)
        # Voltage group: Z2^5 (32 elements)
        
        def construct_320_cage():
            # Base graph vertices 0-4 (A), 5-9 (B)
            # Edges (i, 5+j) for i != j
            base_edges = []
            for i in range(5):
                for j in range(5):
                    if i != j:
                        base_edges.append((i, 5 + j))
            
            # Find all cycles of length < 10 in the base graph
            # We specifically need to check 4-cycles and 6-cycles
            # For Z2^5, we need sum of voltages != 0 for all such cycles
            
            # We can use a simple search: assign random voltages and check
            # Or use Gaussian elimination on the cycle space constraints
            
            # Let's try random assignment first, it's simpler
            num_attempts = 1000
            for _ in range(num_attempts):
                # Assign random 5-bit vectors to edges
                voltages = [random.getrandbits(5) for _ in base_edges]
                
                # Check all 4-cycles
                valid = True
                # A 4-cycle in K5,5 minus matching:
                # i1 - j1 - i2 - j2 - i1
                # Edges: (i1, 5+j1), (i2, 5+j1), (i2, 5+j2), (i1, 5+j2)
                # Note: i1 != j1, i2 != j1, i2 != j2, i1 != j2
                # Also i1 != i2, j1 != j2
                
                for i1 in range(5):
                    for i2 in range(5):
                        if i1 == i2: continue
                        for j1 in range(5):
                            if j1 == i1 or j1 == i2: continue
                            for j2 in range(5):
                                if j2 == i1 or j2 == i2 or j2 == j1: continue
                                # Check if edges exist
                                # (i1, 5+j1), (i2, 5+j1), (i2, 5+j2), (i1, 5+j2)
                                # All exist if i!=j condition holds
                                
                                # Find indices in base_edges
                                e1 = base_edges.index((i1, 5 + j1))
                                e2 = base_edges.index((i2, 5 + j1))
                                e3 = base_edges.index((i2, 5 + j2))
                                e4 = base_edges.index((i1, 5 + j2))
                                
                                # Sum of voltages
                                s = voltages[e1] ^ voltages[e2] ^ voltages[e3] ^ voltages[e4]
                                if s == 0:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid:
                    continue
                
                # Check 6-cycles (more complex, but 4-cycles are the main constraint for K5,5)
                # If 4-cycles are avoided, girth is at least 6.
                # We need girth >= 10. So we must also avoid 6-cycles and 8-cycles.
                # This random search might not be enough.
                
                # Let's assume if we pass 4-cycles, we might pass higher ones?
                # No, we must check all.
                
                # For now, if we find a 4-cycle-free assignment, we build the graph
                # and verify girth.
                
                # Build the lift
                n_lift = 10 * 32
                lift_edges = []
                # Map base vertex and voltage to lift vertex
                # v_lift = v_base * 32 + v_voltage
                
                # We need to determine the voltage of the lift edges
                # The lift edge between (u, x) and (v, x + w) exists if (u,v) is in base with voltage w
                # Wait, the standard voltage graph definition:
                # Edge (u, x) -- (v, x + w_{uv})
                
                for u, v in base_edges:
                    # Find the voltage assigned to this edge
                    # We stored voltages in a list corresponding to base_edges
                    idx = base_edges.index((u, v))
                    w = voltages[idx]
                    
                    for x in range(32):
                        u_lift = u * 32 + x
                        v_lift = v * 32 + ((x + w) % 32)
                        lift_edges.append((u_lift, v_lift))
                        # Also the reverse edge? No, undirected.
                        # But we need to ensure we don't add duplicates.
                        # Since we iterate over all x, we add each undirected edge once?
                        # No, for each base edge, we add 32 lift edges.
                        # Total edges = 20 * 32 = 640.
                        # 320 vertices * 4 / 2 = 640. Correct.
                
                # Verify girth
                if has_girth_at_least(n_lift, lift_edges, 10):
                    return n_lift, lift_edges
            
            return None

        result = construct_320_cage()
        if result:
            return result

        # Fall back to LUW
        d_try = 5
        n, edges = luw_construct(d_try, 4)
        if has_girth_at_least(n, edges, 10):
            return n, edges
            
        d_try = 6
        n, edges = luw_construct(d_try, 4)
        if has_girth_at_least(n, edges, 10):
            return n, edges

    # General case
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = luw_construct(d, k)
    return n, edges
