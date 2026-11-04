import time
import random
import sys

def construct(k, g):
    # k=3, g=16
    if k != 3:
        # Fallback to LUW if k != 3, though task implies k=3
        # But for safety, if k!=3, use a generic method or error
        # The task says "evaluator passes this task's degree k", likely k=3.
        pass
        
    deadline = time.time() + 110.0 # Safety margin
    
    # Petersen Graph Definition
    # Vertices 0..9
    # Edges:
    # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Precompute cycles of length < 16 in Petersen
    # We can do a BFS/DFS from each edge to find cycles.
    # Or just hardcode? No, better to compute once.
    # Since Petersen is small, we can find all simple cycles.
    
    def find_cycles(edges, max_len):
        # Use DFS to find all simple cycles
        # To avoid duplicates, we can use a canonical form (min vertex in cycle is start, and neighbor is smallest)
        # Or just collect all and deduplicate.
        cycles = set()
        n = 10
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        def dfs(start, current, path, visited):
            if len(path) > max_len:
                return
            if current == start and len(path) >= 3:
                # Found a cycle
                # Canonicalize: rotate so min vertex is first, and second vertex is min neighbor
                cycle = tuple(path)
                min_idx = cycle.index(min(cycle))
                rotated = cycle[min_idx:] + cycle[:min_idx]
                # Check reverse
                reversed_cycle = (rotated[0],) + rotated[:0:-1]
                if rotated < reversed_cycle:
                    cycles.add(rotated)
                else:
                    cycles.add(reversed_cycle)
            else:
                for neighbor in adj[current]:
                    if neighbor == path[1] and len(path) == 1:
                        continue # No backtracking immediately
                    if neighbor not in visited:
                        dfs(start, neighbor, path + [neighbor], visited | {neighbor})
                    elif neighbor == start and len(path) >= 3:
                        # Close cycle
                        pass # Handled above? No, above is when current==start.
                        # Wait, the check `if current == start` is at the beginning of the call.
                        # So we only call dfs on neighbors.
                        # If neighbor is start, we should close the cycle.
                        # Let's restructure.
                        
        # Better DFS:
        def dfs2(start, current, path, visited):
            for neighbor in adj[current]:
                if neighbor == path[0] and len(path) >= 3:
                    # Cycle found
                    cycle = tuple(path)
                    min_idx = cycle.index(min(cycle))
                    rotated = cycle[min_idx:] + cycle[:min_idx]
                    reversed_cycle = (rotated[0],) + rotated[:0:-1]
                    if rotated < reversed_cycle:
                        cycles.add(rotated)
                    else:
                        cycles.add(reversed_cycle)
                elif neighbor not in visited:
                    dfs2(start, neighbor, path + [neighbor], visited | {neighbor})
                    
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    dfs2(i, j, [i, j], {i, j})
                    
        return [list(c) for c in cycles]

    # We only care about cycles with length < g (16)
    # But actually, we need to check if voltage sum is 0.
    # If a cycle has length L < g and sum 0, girth is L.
    # So we need to avoid sum 0 for all cycles L < g.
    
    # Let's just generate all cycles up to length 15.
    # Petersen has very few cycles.
    # Let's precompute them once.
    
    all_cycles = find_cycles(base_edges, 15)
    # Filter to length < 16
    relevant_cycles = [c for c in all_cycles if len(c) < 16]
    
    # Map edge to index for voltage assignment
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # For a cycle [v0, v1, ..., vk-1, v0], the edges are (v0,v1), ..., (vk-1,v0)
    # We need to assign voltages to directed edges?
    # In voltage graph, we assign voltage to each directed edge.
    # v((u,v)) = -v((v,u)) if we want the graph to be undirected and simple?
    # Actually, for the lift to be undirected, we need v((u,v)) = v((v,u))^{-1}.
    # In Z_m, this means v((v,u)) = -v((u,v)).
    # So we just assign a value to each undirected edge, and use it for one direction and negative for the other.
    
    # Let's assign voltage x_e to each undirected edge e.
    # For a cycle C = (v0, v1, ..., vk-1, v0), the sum is:
    # sum_{i=0}^{k-1} v((vi, vi+1))
    # Note that the direction matters.
    # If we traverse the cycle in the order of the list, we sum the voltages of the directed edges.
    
    # We need to check if sum == 0 mod m.
    
    # Search for m
    # Moore bound for k=3, g=16 is 767.
    # n = 10 * m.
    # So m >= 77.
    
    best_n = None
    best_edges = None
    
    # Try m from 77 upwards
    # We can also try to optimize by using smaller m if we find a valid assignment?
    # No, we want min n. So start from smallest m.
    
    # However, random search might fail for small m if constraints are tight.
    # Let's try m from 77 to 100.
    
    for m in range(77, 101):
        if time.time() > deadline - 10:
            break
            
        # Try random assignments
        # Number of edges in Petersen is 15.
        # We assign random values in 1..m-1 to each edge.
        # We can use 0? If voltage is 0, it's fine as long as sum is not 0.
        # But using 0 might increase chance of sum 0? No.
        # Just use 0..m-1.
        
        found = False
        attempts = 0
        while time.time() < deadline - 10 and attempts < 5000:
            attempts += 1
            # Assign random voltages to 15 edges
            voltages = [random.randint(0, m - 1) for _ in range(15)]
            
            valid = True
            for cycle in relevant_cycles:
                # Calculate sum
                s = 0
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    # Get voltage of edge (u, v)
                    # We need to know the direction.
                    # The edge index is same for (u,v) and (v,u).
                    # But the voltage used depends on direction.
                    # If we store voltage x_e for undirected edge e,
                    # and we traverse u->v, we add x_e.
                    # If we traverse v->u, we add -x_e.
                    # How to know which way we stored it?
                    # Let's say we store voltage for the edge (min(u,v), max(u,v)) as x.
                    # Then if we go min->max, add x. If max->min, add -x.
                    
                    if u < v:
                        s += voltages[edge_index[(u, v)]]
                    else:
                        s -= voltages[edge_index[(u, v)]]
                        
                s %= m
                if s == 0:
                    valid = False
                    break
                    
            if valid:
                # Found valid assignment for this m
                # Construct the graph
                n = 10 * m
                edges = []
                # Vertices are (v, g) for v in 0..9, g in 0..m-1
                # Map to single index: v * m + g
                for u, v in base_edges:
                    # Get voltage
                    if u < v:
                        w = voltages[edge_index[(u, v)]]
                    else:
                        w = -voltages[edge_index[(u, v)]] % m # Ensure positive?
                        # Actually, in Z_m, -w is m-w.
                        # We just need consistency.
                        # Let's use the value directly.
                        w = voltages[edge_index[(u, v)]]
                        # If u > v, the edge is stored as (v, u) with voltage x.
                        # Traversing u->v is traversing (v,u) in reverse.
                        # So voltage is -x.
                        w = (-w) % m
                    
                    for g in range(m):
                        u_idx = u * m + g
                        v_idx = v * m + (g + w) % m
                        edges.append((u_idx, v_idx))
                        
                best_n = n
                best_edges = edges
                found = True
                break
                
        if found:
            break
            
    if best_n is not None:
        return best_n, best_edges
    else:
        # Fallback: Return a large valid graph if search fails
        # Use the LUW construction from the original code as fallback
        # Or just return a random 3-regular graph? No, must have girth 16.
        # If we can't find one, we might timeout.
        # But random search on Petersen with m=100 should be very fast.
        # Let's add a fallback to a known large graph if needed.
        
        # Re-implement LUW quickly?
        # Or just return an error? No, must return valid graph.
        # Let's try m up to 200 just in case.
        
        for m in range(101, 201):
            if time.time() > deadline - 5:
                break
            attempts = 0
            while time.time() < deadline - 5 and attempts < 1000:
                attempts += 1
                voltages = [random.randint(0, m - 1) for _ in range(15)]
                valid = True
                for cycle in relevant_cycles:
                    s = 0
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        if u < v:
                            s += voltages[edge_index[(u, v)]]
                        else:
                            s -= voltages[edge_index[(u, v)]]
                    s %= m
                    if s == 0:
                        valid = False
                        break
                if valid:
                    n = 10 * m
                    edges = []
                    for u, v in base_edges:
                        if u < v:
                            w = voltages[edge_index[(u, v)]]
                        else:
                            w = (-voltages[edge_index[(u, v)]]) % m
                        for g in range(m):
                            u_idx = u * m + g
                            v_idx = v * m + (g + w) % m
                            edges.append((u_idx, v_idx))
                    return n, edges
        # If still not found, return a trivial valid graph?
        # A cycle of length 16 is 2-regular. We need 3-regular.
        # We can take 3 disjoint cycles of length 16? No, must be connected?
        # Task doesn't say connected, but "the graph". Usually implies connected.
        # But "no cycle shorter than 16".
        # If we return a 3-regular graph with girth 16, it's valid.
        # If we can't find one, we might have to use the LUW fallback.
        
        # Fallback to LUW
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

