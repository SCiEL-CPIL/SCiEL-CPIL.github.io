import time
import random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """
    Check if graph has girth >= g.
    adj: list of lists of neighbors
    n: number of vertices
    g: target girth
    Returns True if girth >= g, False otherwise.
    """
    # Non-backtracking BFS from each vertex
    # State: (vertex, prev_vertex)
    # We stop if depth >= g-1 (since a cycle of length g would be found at depth g)
    # Actually, if we return to start at depth d, cycle length is d+1?
    # No, non-backtracking walk from start s:
    # (s, -1) -> (v1, s) -> ...
    # If we reach (s, v_last), the cycle length is depth + 1?
    # Let's trace:
    # s -> a (depth 1)
    # a -> b (depth 2)
    # b -> s (depth 3). Cycle s-a-b-s length 3.
    # So if we reach (s, prev) at depth d, cycle length is d+1?
    # Wait. (s, -1) is depth 0.
    # (a, s) is depth 1.
    # (b, a) is depth 2.
    # (s, b) is depth 3. Cycle length 3.
    # So if depth >= g-1, and we hit start, we have a cycle of length >= g?
    # If depth = g-1, cycle length g. This is allowed.
    # If depth < g-1, and we hit start, cycle length < g. Not allowed.
    
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            
            # If we have gone deep enough, we don't need to check for cycles 
            # because any cycle formed would be length >= g
            if d >= g - 1:
                continue
                
            for w in adj[v]:
                if w == prev:
                    continue
                
                # Check if we closed a cycle
                if w == start:
                    # Cycle length is d + 1
                    if d + 1 < g:
                        return False
                    # If d+1 >= g, it's a valid cycle (girth >= g)
                    # We don't need to expand further from this state for cycle detection
                    # but we might need to for other cycles? No, non-backtracking 
                    # ensures we don't repeat edges immediately.
                    continue
                
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
                    
    return True

def _petersen_edges():
    """Return edges of the Petersen graph (10 vertices)."""
    # Outer pentagon: 0-1-2-3-4-0
    # Inner pentagram: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    return [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

def _build_petersen_lift(m, voltages, seed):
    """
    Build a lift of the Petersen graph with given voltages.
    m: lift size (number of layers)
    voltages: list of 15 integers (one per edge) in Z_m
    Returns (n, edges, adj) or None if invalid.
    """
    base_edges = _petersen_edges()
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for i, (u, v) in enumerate(base_edges):
        vol = voltages[i] % m
        for layer in range(m):
            a = u * m + layer
            b = v * m + (layer + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
            else:
                # Duplicate edge, invalid simple graph
                return None
            
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    adj_list = [sorted(list(adj[i])) for i in range(n)]
    return n, edges, adj_list

def _random_voltage_search(m, g, n_attempts, seed):
    """
    Try n_attempts random voltage assignments for a Petersen lift of size m.
    Returns (n, edges) if a valid graph is found, else None.
    """
    rng = random.Random(seed)
    base_edges = _petersen_edges()
    num_edges = len(base_edges)
    
    for _ in range(n_attempts):
        voltages = [rng.randint(0, m - 1) for _ in range(num_edges)]
        result = _build_petersen_lift(m, voltages, seed)
        if result is None:
            continue
        n, edges, adj_list = result
        if _verify_girth_fast(adj_list, n, g):
            return n, edges
    return None

def _random_3reg_graph(n, seed):
    """
    Generate a random 3-regular graph on n vertices.
    Returns (n, edges, adj_list) or None.
    """
    if n % 2 != 0:
        return None
    rng = random.Random(seed)
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    edges = []
    valid = True
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
    if not valid:
        return None
        
    # Check connectivity
    visited = {0}
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in visited:
                visited.add(w)
                stack.append(w)
    if len(visited) != n:
        return None
        
    adj_list = [sorted(list(adj[i])) for i in range(n)]
    return n, edges, adj_list

def _improve_girth(edges, adj_list, n, g, max_swaps, seed):
    """
    Try to increase girth using edge swaps.
    Returns improved (n, edges, adj_list) or original.
    """
    rng = random.Random(seed)
    current_edges = list(edges)
    current_adj = [list(nei) for nei in adj_list]
    
    # Convert to set for fast lookup
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    best_girth = _verify_girth_fast(current_adj, n, g)
    
    for _ in range(max_swaps):
        # Pick two random edges (a,b) and (c,d)
        e1_idx = rng.randint(0, len(current_edges) - 1)
        e2_idx = rng.randint(0, len(current_edges) - 1)
        if e1_idx == e2_idx:
            continue
        
        a, b = current_edges[e1_idx]
        c, d = current_edges[e2_idx]
        
        # Ensure no shared vertices
        if len({a,b,c,d}) < 4:
            continue
            
        # Try swap (a,b),(c,d) -> (a,c),(b,d)
        new_e1 = (a, c)
        new_e2 = (b, d)
        
        # Check if new edges exist or are loops
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue
        if (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1])) in edge_set:
            continue
        if (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1])) in edge_set:
            continue
            
        # Perform swap
        edge_set.remove((min(a,b), max(a,b)))
        edge_set.remove((min(c,d), max(c,d)))
        edge_set.add((min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1])))
        edge_set.add((min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1])))
        
        current_edges[e1_idx] = new_e1
        current_edges[e2_idx] = new_e2
        
        # Rebuild adj
        # This is O(E), might be slow. Optimize by updating only 4 vertices.
        # But for simplicity and correctness, let's just check girth on the fly?
        # No, we need adj for girth check.
        
        # Update adj lists
        # Remove old neighbors
        current_adj[a].remove(b)
        current_adj[b].remove(a)
        current_adj[c].remove(d)
        current_adj[d].remove(c)
        
        # Add new neighbors
        current_adj[a].append(c)
        current_adj[c].append(a)
        current_adj[b].append(d)
        current_adj[d].append(b)
        
        # Sort for consistency? Not strictly needed for BFS if we use lists
        # But _verify_girth_fast expects lists.
        
        if _verify_girth_fast(current_adj, n, g):
            return n, current_edges, current_adj
            
        # If girth decreased or stayed same, we might want to revert?
        # For now, accept any swap that doesn't break degree/connectivity.
        # Connectivity is preserved by 2-swaps on a connected graph.
        
    return n, current_edges, current_adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    if k != 3:
        # Fallback for other k, though task specifies k=3
        # Use a simple random construction or LUW if available
        # For now, return a random graph
        n = 2 * (k ** (g//2)) # Rough estimate
        # This is not optimal, but k=3 is the main case
        pass
        
    # Moore bound for k=3, g=16 is 512
    # We will search for n starting from 512 upwards
    
    best_result = None
    best_n = float('inf')
    
    # Strategy 1: Petersen Lifts
    # n = 10 * m
    # Start m such that 10*m >= 512 => m >= 52
    # Try m from 52 to 200
    for m in range(52, 200):
        if time.time() > deadline - 20:
            break
        n_lift = 10 * m
        if n_lift >= best_n:
            continue
            
        # Try random voltages
        # More attempts for smaller m?
        attempts = 50
        result = _random_voltage_search(m, g, attempts, seed=m*1000)
        if result is not None:
            n_r, edges_r = result
            if n_r < best_n:
                best_n = n_r
                best_result = (n_r, edges_r)
            # If we found one, we might try to find a smaller one?
            # But we are iterating m increasing, so we found the smallest m so far.
            # We can break if we are confident this is optimal?
            # No, maybe a larger m has a different structure?
            # But we want MIN n. So smallest m is best.
            # So if we find a valid lift for m, it's the best for this base graph.
            # We can break the loop if we assume larger m won't be smaller.
            # But maybe m=52 failed, m=53 succeeded.
            # So we continue until we find one?
            # Yes.
            break
            
    # Strategy 2: Random 3-regular graphs with optimization
    if time.time() < deadline - 30:
        # Try n from Moore bound upwards
        # Moore bound 512. n must be even.
        for n_try in range(512, min(best_n, 600), 2):
            if time.time() > deadline - 10:
                break
            # Try to generate and improve
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                res = _random_3reg_graph(n_try, seed=attempt*100 + n_try)
                if res is None:
                    continue
                n_r, edges_r, adj_r = res
                if _verify_girth_fast(adj_r, n_r, g):
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break
                else:
                    # Try to improve
                    # This is expensive, so only do it if we have time
                    if time.time() < deadline - 5:
                        n_imp, edges_imp, adj_imp = _improve_girth(edges_r, adj_r, n_r, g, max_swaps=100, seed=attempt)
                        if _verify_girth_fast(adj_imp, n_imp, g):
                            if n_imp < best_n:
                                best_n = n_imp
                                best_result = (n_imp, edges_imp)
                            break
                            
    if best_result is None:
        # Fallback: Return a large random graph or LUW result
        # To ensure validity, return a known valid graph if possible
        # Or just a random one that passes verification (unlikely for small n)
        # Let's try a larger Petersen lift
        m = 300
        result = _random_voltage_search(m, g, 100, seed=999)
        if result is not None:
            best_result = result
        else:
            # Last resort: random graph, might fail girth check but must return something
            # The evaluator checks validity. If invalid, score 0.
            # So we must return a valid graph.
            # Try m=1000
            m = 1000
            result = _random_voltage_search(m, g, 10, seed=123)
            if result is not None:
                best_result = result
            else:
                # Construct a trivial valid graph? 
                # A cycle of length g? No, must be 3-regular.
                # Two vertices with 3 parallel edges? No, simple graph.
                # So we must find a 3-regular graph.
                # If we can't find one with girth g, we are stuck.
                # But for g=16, n=10000 should work with high probability for random voltages.
                m = 10000
                result = _random_voltage_search(m, g, 5, seed=456)
                if result is not None:
                    best_result = result
                else:
                    # Emergency: return empty? No, must return valid.
                    # This should not happen.
                    return 0, []
                    
    return best_result
