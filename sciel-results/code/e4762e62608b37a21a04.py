<<<<<<< SEARCH
import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 5 or g != 9:
        # Fallback to LUW for other cases, though task is specific
        return _luw_construct(k, g, deadline)
    
    # Target N is around Moore bound 426
    # We will search for voltage graph lifts
    
    # Generate candidate base graphs
    base_graphs = []
    
    # 1. K6
    edges_k6 = [(i, j) for i in range(6) for j in range(i+1, 6)]
    base_graphs.append((6, edges_k6))
    
    # 2. K5,5
    edges_k55 = [(i, j+5) for i in range(5) for j in range(5)]
    base_graphs.append((10, edges_k55))
    
    # 3. Random 5-regular graphs on V vertices
    # V must be even for regular graph? No, sum of degrees = 5V must be even, so V must be even.
    for V in [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]:
        # Try to generate a random 5-regular graph
        # Using configuration model (pairing stubs)
        # Stubs: 5 per vertex
        stubs = [i for i in range(V) for _ in range(5)]
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
        if valid and len(edges) == 5 * V // 2:
            base_graphs.append((V, list(edges)))
            
    # Search for voltage assignments
    best_n = float('inf')
    best_edges = None
    
    # Prioritize smaller V (smaller base, larger m) or specific known good ones?
    # Smaller V means larger m, which might make voltage sums harder to be 0?
    # Actually, larger m makes it easier for sums to be non-zero.
    # So K6 (V=6, m~70) is a good candidate.
    
    # Sort base graphs by V
    base_graphs.sort(key=lambda x: x[0])
    
    for V, base_edges in base_graphs:
        if time.time() > deadline - 10:
            break
            
        # Determine m
        # We want N = V * m to be close to 426
        # m should be an integer.
        # Try m in range [426//V - 5, 426//V + 5]
        target_m = 426 // V
        for m in range(max(2, target_m - 5), target_m + 6):
            N = V * m
            if N < 426: # Moore bound is 426, so N must be >= 426? 
                # Actually, Moore bound is a lower bound. So N >= 426.
                # If N < 426, it's impossible to have girth 9?
                # Yes, if N < Moore bound, girth must be less than 9.
                continue
                
            # Try a few random voltage assignments
            for trial in range(20): # 20 trials per (V, m)
                if time.time() > deadline - 5:
                    break
                
                # Assign random voltages in Z_m
                voltages = {}
                for u, v in base_edges:
                    voltages[(u, v)] = random.randint(0, m-1)
                    voltages[(v, u)] = (m - voltages[(u, v)]) % m # Inverse for reverse direction?
                    # In voltage graphs, usually we define voltage for directed edge.
                    # If we use undirected edges, we need consistency.
                    # Let's define voltage for (u,v) as w. Then (v,u) is -w.
                    
                # Build the lift graph
                # Vertices are (v, i) where v in 0..V-1, i in 0..m-1
                # Edge between (u, i) and (v, j) if (u,v) in base and j = (i + voltage(u,v)) % m
                lift_adj = [[] for _ in range(N)]
                
                for u, v in base_edges:
                    w = voltages[(u, v)]
                    # u -> v
                    for i in range(m):
                        u_idx = u * m + i
                        v_idx = v * m + (i + w) % m
                        lift_adj[u_idx].append(v_idx)
                        lift_adj[v_idx].append(u_idx)
                        
                # Check girth
                if _check_girth(lift_adj, N, g):
                    # Extract edges
                    edges = []
                    for u in range(N):
                        for v in lift_adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if N < best_n:
                        best_n = N
                        best_edges = edges
                        # If we found a very good one, we can stop early?
                        # Maybe continue to find smaller?
                        # But 426 is the bound. If we find 426, we can't do better.
                        if N == 426:
                            return N, edges
                            
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to LUW
    return _luw_construct(k, g, deadline)
=======
import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 105  # Safety margin before 150s limit
    
    if k != 5 or g != 9:
        return _luw_construct(k, g, deadline)
    
    # Try voltage graph approach with strict time budget
    best_n, best_edges = _voltage_graph_search(k, g, deadline)
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW with remaining time
    return _luw_construct(k, g, deadline)

def _voltage_graph_search(k, g, deadline):
    """Search for voltage graph lifts with tight time budget."""
    start_time = time.time()
    time_budget = 30  # Give at most 30s to voltage graph search
    
    base_graphs = []
    
    # K6
    edges_k6 = [(i, j) for i in range(6) for j in range(i+1, 6)]
    base_graphs.append((6, edges_k6))
    
    # K5,5
    edges_k55 = [(i, j+5) for i in range(5) for j in range(5)]
    base_graphs.append((10, edges_k55))
    
    # Random 5-regular graphs
    for V in [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]:
        stubs = [i for i in range(V) for _ in range(5)]
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v or (min(u,v), max(u,v)) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
        if valid and len(edges) == 5 * V // 2:
            base_graphs.append((V, list(edges)))
    
    best_n = float('inf')
    best_edges = None
    
    base_graphs.sort(key=lambda x: x[0])
    
    for V, base_edges in base_graphs:
        if time.time() - start_time > time_budget:
            break
            
        target_m = 426 // V
        for m in range(max(2, target_m - 3), target_m + 4):
            N = V * m
            if N < 426:
                continue
                
            # Try fewer trials with faster check
            for trial in range(5):
                if time.time() - start_time > time_budget:
                    break
                
                voltages = {}
                for u, v in base_edges:
                    w = random.randint(0, m-1)
                    voltages[(u, v)] = w
                    voltages[(v, u)] = (-w) % m
                
                # Build adjacency as list of lists for fast access
                adj = [[] for _ in range(N)]
                for u, v in base_edges:
                    w = voltages[(u, v)]
                    for i in range(m):
                        u_idx = u * m + i
                        v_idx = v * m + (i + w) % m
                        adj[u_idx].append(v_idx)
                        adj[v_idx].append(u_idx)
                
                if _fast_girth_check(adj, N, g):
                    edges = []
                    for u in range(N):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if N < best_n:
                        best_n = N
                        best_edges = edges
                        if N == 426:
                            return N, edges
    
    return best_n, best_edges

def _fast_girth_check(adj, n, min_girth):
    """Fast girth check using BFS with early termination."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= min_girth - 1:
                continue  # No shorter cycle possible from here
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True
>>>>>>> REPLACE

