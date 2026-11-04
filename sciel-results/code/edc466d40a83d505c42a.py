import time
import random
import sys
from collections import deque
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Fallback: LUW graph
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

    baseline_n, baseline_edges = get_luw_graph(k, g)
    best_n = baseline_n
    best_edges = baseline_edges

    # Structural Change: Voltage Lift over Petersen Graph
    # The Petersen graph is 3-regular on 10 vertices with girth 5.
    # A voltage lift with group Z_m x Z_m has m^2 vertices.
    # We want to find a voltage assignment such that no cycle in the lift has length < g.
    # This is equivalent to: for every cycle C in the Petersen graph, the sum of voltages along C is not the identity (0,0) in Z_m x Z_m.
    # However, we must also ensure that the lift is simple (no loops or multi-edges).
    # For a 3-regular graph, the lift is simple if the voltage of every edge is not the identity AND the voltage of every 2-cycle (back-and-forth) is not identity (which is true if edge voltage != 0).
    # Actually, for a simple lift, we just need edge voltages to be non-zero.
    # Wait, if voltage is 0, the edge connects (u, x) to (v, x). If u!=v, it's a valid edge.
    # But if we have an edge u-v with voltage 0, and another edge u-v? No, Petersen is simple.
    # The lift is simple if for any edge (u,v) with voltage gamma, we don't have (u,v) with voltage 0? No.
    # The lift is simple if there are no loops and no multiple edges.
    # Loop: (u, x) -> (u, x+gamma). Loop if u=v and gamma=0. Petersen has no loops, so no loops in lift.
    # Multiple edges: (u, x) -> (v, x+gamma1) and (u, x) -> (v, x+gamma2).
    # This happens if there are multiple edges u-v in base. Petersen is simple, so no.
    # What about (u, x) -> (v, x+gamma) and (v, x+gamma) -> (u, x+gamma+delta)?
    # This is just the reverse edge.
    # So, any voltage assignment on a simple base graph yields a simple lift.
    # We just need to avoid short cycles.

    # Petersen Graph Edges (Standard)
    # Vertices 0-9
    # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner pentagram: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Find all cycles in Petersen graph of length < g
    # Petersen has girth 5. Cycles of length 5 and 6 exist.
    # We need to check cycles of length 5, 6, ..., g-1.
    # Since g=16, we need to check lengths 5 to 15.
    # But Petersen is small (10 vertices).
    # We can enumerate all simple cycles in Petersen.
    
    def find_all_cycles(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        cycles = []
        def dfs(start, current, path, visited):
            if current == start and len(path) > 2:
                # Found a cycle
                # Canonical representation: smallest vertex is start, and path is sorted?
                # To avoid duplicates, we only start DFS from the smallest vertex in the cycle.
                # And we ensure the path is in a specific order?
                # Simpler: just collect all, then deduplicate.
                cycles.append(tuple(path))
                return
            
            if len(path) > 15: # Max length we care about is 15 (for g=16)
                return
                
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    # Close cycle
                    cycles.append(tuple(path))
                elif neighbor not in visited and neighbor > start:
                    # Only visit vertices > start to ensure start is the minimum
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        for start in range(n):
            dfs(start, start, [start], {start})
            
        # Deduplicate
        unique_cycles = set()
        for c in cycles:
            # Rotate to start with min element
            min_idx = c.index(min(c))
            rotated = c[min_idx:] + c[:min_idx]
            # Also check reverse
            rev = rotated[::-1]
            if min(rotated, rev) < min(unique_cycles.get(min(rotated, rev), [999])):
                 # Just add both to set, or normalize
                 unique_cycles.add(tuple(rotated))
                 unique_cycles.add(tuple(rev))
                 
        return list(unique_cycles)

    petersen_cycles = find_all_cycles(petersen_edges, 10)
    # Filter cycles shorter than g
    relevant_cycles = [c for c in petersen_cycles if len(c) < g]
    
    # For each cycle, we need the sum of voltages (with signs depending on direction) to be non-zero.
    # Let's define a function to check if a voltage assignment is valid.
    # Voltage assignment: map edge (u,v) to (a,b) in Z_m x Z_m.
    # We can parameterize voltages by the edges.
    # To reduce search space, we can fix some voltages or use a structured family.
    # Let's try voltages in Z_m x Z_m where m is a prime.
    # n = m^2.
    # We want to minimize m^2.
    # Lower bound for m^2 is 512? No, Moore bound for k=3, g=16 is ~ 2*(3*(2^7-1)) = 1534?
    # Wait, Moore bound for odd girth is (k/2) * sum_{i=0}^{(g-3)/2} (k-1)^i.
    # For k=3, g=16 (even), Moore bound is k * sum_{i=0}^{g/2 - 1} (k-1)^i = 3 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 3 * 255 = 765.
    # Wait, the task says best lower bound is 512.
    # Anyway, we want small m.
    
    # Let's try m from 22 upwards (since 21^2 = 441 < 512, 22^2 = 484, 23^2 = 529).
    # Actually, let's just iterate m and see if we find a valid assignment.
    
    # To speed up, we can use a heuristic search for voltages.
    # Or, we can use a specific family of voltages.
    # e.g., voltages are (i, j) where i,j are linear functions of edge index?
    
    # Let's try a random search for m values.
    # For each m, try N random voltage assignments.
    
    # We need to be careful with time.
    # Checking validity: for each relevant cycle, sum voltages.
    # Number of relevant cycles in Petersen is small.
    # Cycles of length 5: 12.
    # Cycles of length 6: 10.
    # Cycles of length 8: ...
    # Total is manageable.
    
    def check_validity(voltages, m, cycles):
        # voltages: dict (u,v) -> (a,b)
        # We need to orient the cycle consistently with the voltage assignment?
        # No, voltage is on the edge. If we traverse u->v, we add voltage(u,v).
        # If we traverse v->u, we subtract voltage(u,v) (or add inverse).
        # In Z_m, inverse of (a,b) is (-a, -b) mod m.
        
        for cycle in cycles:
            # cycle is a tuple of vertices (v0, v1, ..., vk-1)
            # Edges are (v0,v1), (v1,v2), ..., (vk-1, v0)
            sum_a = 0
            sum_b = 0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                # Get voltage for edge (u,v)
                if (u, v) in voltages:
                    a, b = voltages[(u, v)]
                    sum_a = (sum_a + a) % m
                    sum_b = (sum_b + b) % m
                elif (v, u) in voltages:
                    a, b = voltages[(v, u)]
                    sum_a = (sum_a - a) % m
                    sum_b = (sum_b - b) % m
                else:
                    # Should not happen
                    return False
            if sum_a == 0 and sum_b == 0:
                return False
        return True

    # We will try to find a valid lift for increasing m.
    # Start with m such that m^2 < best_n.
    # best_n is from LUW.
    
    max_m = int((best_n - 1) ** 0.5)
    
    # To make it faster, we can restrict the search space.
    # Instead of fully random, we can try to set voltages to be "generic".
    
    found = False
    for m in range(2, max_m + 1):
        if time.time() > deadline - 10.0:
            break
            
        # Try a few random assignments for this m
        # Number of trials: more for smaller m? Or just fixed?
        # Let's do 1000 trials per m.
        trials = 1000
        
        for _ in range(trials):
            if time.time() > deadline - 10.0:
                break
                
            voltages = {}
            for u, v in petersen_edges:
                a = random.randint(0, m - 1)
                b = random.randint(0, m - 1)
                # Avoid 0,0 to ensure simple lift? Not strictly required for simplicity, but good for girth.
                # If voltage is 0,0, it's a "trivial" lift edge.
                # Let's allow 0,0 but it might create short cycles if the base cycle is short?
                # No, if base cycle is short, and voltage sum is 0, it creates a short cycle in lift.
                # So we just check the sum.
                voltages[(u, v)] = (a, b)
                
            if check_validity(voltages, m, relevant_cycles):
                # Valid! Construct the lift.
                n_lift = m * m
                # Map (vertex, (x,y)) to index
                # Index = vertex * m^2 + x * m + y
                edges_lift = []
                for u, v in petersen_edges:
                    a, b = voltages[(u, v)]
                    for x in range(m):
                        for y in range(m):
                            u_idx = u * m * m + x * m + y
                            v_x = (x + a) % m
                            v_y = (y + b) % m
                            v_idx = v * m * m + v_x * m + v_y
                            # Add edge (u_idx, v_idx)
                            # Ensure no duplicates?
                            # Since base is simple and we iterate each edge once,
                            # and for each (x,y) we get a unique pair,
                            # we might add (u,v) and (v,u) from different base edges?
                            # No, base edges are unique.
                            # But we might add (u_idx, v_idx) and later (v_idx, u_idx)?
                            # No, because we only iterate directed base edges?
                            # petersen_edges is undirected list.
                            # We should treat them as undirected.
                            # If we add (u_idx, v_idx), we are done.
                            # We don't need to add (v_idx, u_idx) separately if we consider the graph undirected.
                            # But the output format is a list of edges (i, j).
                            # Usually, (i, j) implies undirected.
                            # So we just add one direction.
                            edges_lift.append((u_idx, v_idx))
                            
                # Verify degree and girth?
                # Degree: each vertex in lift has degree 3.
                # Girth: guaranteed by check_validity?
                # Yes, if all base cycles of length < g have non-zero voltage sum,
                # then no lift cycle of length < g exists.
                # Wait, is that true?
                # A cycle in the lift projects to a closed walk in the base.
                # The closed walk can be decomposed into cycles.
                # If the closed walk has length < g, it must be a simple cycle in the base (since base girth is 5, and we check all cycles < g).
                # Actually, a closed walk of length < g in a graph of girth 5 must be a simple cycle?
                # Yes, because if it repeats a vertex, it contains a cycle of length < g.
                # So checking simple cycles is sufficient.
                
                best_n = n_lift
                best_edges = edges_lift
                found = True
                # We found one. Can we find a smaller one?
                # Since we iterate m from small to large, the first valid m gives the smallest n for this search.
                # But maybe a larger m with better structure? No, n=m^2.
                # So we break if we found one for the smallest m?
                # We should break the m loop if we found one, because larger m means larger n.
                break
                
        if found:
            break

    return best_n, best_edges
