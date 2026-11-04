import random
import time
import math

def construct(k, g):
    # Target: 3-regular, girth >= g (16)
    # Base: Petersen Graph
    # Lift: Voltage group Z_m
    
    if k != 3:
        # Fallback to LUW for other k? 
        # The task specifies k=3, g=16. 
        # But construct(k, g) is the interface. 
        # If k != 3, we might need a different strategy or just fail/return large.
        # The prompt says "evaluator passes this task's degree k and girth g". 
        # So k=3, g=16.
        # We will implement the Petersen lift for k=3.
        # For other k, we can fall back to the seed logic or a simple construction.
        # Given the specific task, we assume k=3.
        pass

    # Petersen Graph
    # Vertices 0..9
    # Outer cycle: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Generate closed non-backtracking walks of length < g
    # A walk is a sequence of (edge_id, direction)
    # direction: 1 if u->v matches edges[e], -1 if v->u
    # We want to find all closed NB walks of length L where 3 <= L < g
    # (Length 1, 2 are not cycles in simple graph or backtracking)
    
    walks = []
    
    # To avoid duplicates, we can just generate all and deduplicate?
    # Or just check all.
    # Number of walks is small.
    
    for start_v in range(n_base):
        # BFS state: (current_v, prev_edge, length, path)
        # path is list of (edge_id, sign)
        queue = []
        # Initial step: take any edge from start_v
        for next_v, e_id in adj[start_v]:
            # sign: 1 if start_v is u in edges[e_id], else -1
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
            
        while queue:
            curr_v, prev_e, length, path = queue.pop(0)
            
            if length >= g:
                continue
                
            # Check if closed
            if curr_v == start_v and length >= 3:
                # It is a closed walk
                # We add it to walks
                # To reduce redundancy, we can normalize? 
                # Or just keep all. 
                # Duplicates will just be checked multiple times, which is fine for random search.
                walks.append(path)
                
            # Extend
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue # Non-backtracking
                u, v = edges[e_id]
                # Determine sign for traversal curr_v -> next_v
                if u == curr_v:
                    sign = 1
                else:
                    sign = -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
                
    # Deduplicate walks? 
    # Two walks are equivalent if they are the same sequence of edges with signs.
    # We can use a set of tuples.
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    walks = list(unique_walks)
    
    # If no walks found (shouldn't happen for Petersen), fallback
    if not walks:
        # Just return a large graph?
        # Or use LUW
        pass

    # Search for m
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Moore bound for k=3, g=16 is 512.
    # n = 10 * m.
    # We want 10 * m >= 512 => m >= 51.2
    # So try m from 52 upwards.
    
    best_n = float('inf')
    best_edges = None
    best_m = None
    
    # Try m values
    # We can try a range of m.
    # If m is prime, it's easier to reason about, but not required.
    
    # Strategy: 
    # 1. Try m in [52, 60, 70, 80, ...]
    # 2. For each m, try random assignments.
    
    # To be safe, we can also try to construct a specific known good graph if random fails?
    # But random should work for small m.
    
    # Pre-compute coefficients for each walk
    # walk is list of (e_id, sign)
    # We can convert to a dict or list of (e_id, coeff)
    # coeff is sign.
    
    walk_coeffs = []
    for w in walks:
        coeffs = [0] * len(edges)
        for e_id, sign in w:
            coeffs[e_id] += sign
        # If a walk uses an edge multiple times with same sign, coeff > 1.
        # This is fine.
        walk_coeffs.append(coeffs)
        
    # Optimization: 
    # If a walk has all coeffs 0, it's always 0. (Shouldn't happen for NB walks in simple graph?)
    # Actually, if a walk goes e1 then -e1, it's backtracking, which is excluded.
    # So coeffs should not be trivially zero.
    
    for m in range(52, 200):
        if time.time() > deadline - 5.0:
            break
            
        # Check if 10*m is a candidate
        candidate_n = 10 * m
        
        # Try random assignments
        found = False
        # Number of tries
        # If m is small, more tries? 
        # Let's do 2000 tries per m.
        for _ in range(2000):
            if time.time() > deadline - 2.0:
                break
                
            # Random voltages
            # Ensure connectivity: gcd(v_1, ..., v_m, m) = 1
            # Just pick random 1..m-1
            volts = [random.randint(1, m-1) for _ in range(len(edges))]
            
            # Check connectivity
            # Compute gcd of all volts and m
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
            if g_val != 1:
                # Not connected (or not full group)
                # Actually, for Z_m, the subgroup generated by volts has size m/gcd(volts, m).
                # If gcd != 1, the lift is disconnected.
                # We require connected graph.
                continue
                
            # Check girth
            valid = True
            for coeffs in walk_coeffs:
                s = 0
                for i, c in enumerate(coeffs):
                    if c != 0:
                        s += c * volts[i]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                found = True
                break
        
        if found:
            best_m = m
            best_n = candidate_n
            # Construct edges
            # Vertices: (v, t) where v in 0..9, t in 0..m-1
            # Edge (u, v) in base with voltage vol:
            # (u, t) -- (v, (t + vol) % m)
            # Note: If we traverse u->v, we add vol. If v->u, we add -vol (or m-vol).
            # Our edge definition in base is undirected.
            # In the lift, each base edge (u,v) with voltage vol becomes:
            # For each t:
            #   Edge between (u, t) and (v, (t + vol) % m)
            # This is a single edge in the lift.
            # Wait, is it?
            # If we have edge (u,v) in base, it connects u and v.
            # In the lift, (u, t) is connected to (v, t+vol).
            # And (v, t) is connected to (u, t-vol).
            # These are the same set of edges.
            # So we just iterate t from 0 to m-1.
            
            lift_edges = []
            for i, (u, v) in enumerate(edges):
                vol = volts[i]
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + (t + vol) % m
                    # Ensure u_lift < v_lift? No, just add.
                    # But we must avoid duplicates?
                    # The edge (u,t)-(v,t+vol) is distinct from (v,t)-(u,t-vol) for a fixed t?
                    # No, the set of edges generated by iterating t for a fixed base edge (u,v) 
                    # is exactly the set of edges in the lift corresponding to that base edge.
                    # Each pair (u_lift, v_lift) is unique for a given base edge and t.
                    # Different base edges produce different sets of edges.
                    # So no duplicates.
                    lift_edges.append((u_lift, v_lift))
                    
            best_edges = lift_edges
            # Since we are searching m from low to high, the first valid one is optimal for this strategy.
            # We can break? 
            # Yes, smaller m means smaller n.
            break
            
    if best_edges is None:
        # Fallback to LUW if random lift failed
        # This ensures we return *something* valid, even if large.
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
        index, order, edges_luw = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_luw.append((index[vertex], index[other]))
        return len(order), edges_luw

    return best_n, best_edges
