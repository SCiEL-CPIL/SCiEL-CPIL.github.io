<<<<<<< SEARCH
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k == 3 and g == 16:
        # Base graph: Cube Q3
        # Vertices 0..7
        base_edges = []
        for u in range(8):
            for v in range(u + 1, 8):
                if bin(u ^ v).count('1') == 1:
                    base_edges.append((u, v))
        
        group_size = 117
        n_lifted = 8 * group_size
        
        # Helper to check girth
        def check_girth(n, edges, target_girth):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check degree
            for i in range(n):
                if len(adj[i]) != k:
                    return False
            
            # BFS from each vertex
            max_depth = target_girth // 2
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            if v != parent[u]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < target_girth:
                                    return False
            return True

        # Random voltage search
        for attempt in range(5000):
            if time.time() > deadline:
                break
                
            voltages = {}
            for u, v in base_edges:
                vu = random.randint(0, group_size - 1)
                voltages[(u, v)] = vu
                voltages[(v, u)] = (-vu) % group_size
            
            lifted_edges = []
            for u, v in base_edges:
                vu = voltages[(u, v)]
                for x in range(group_size):
                    y = (x + vu) % group_size
                    u_lift = u * group_size + x
                    v_lift = v * group_size + y
                    lifted_edges.append((u_lift, v_lift))
            
            if check_girth(n_lifted, lifted_edges, g):
                return n_lifted, lifted_edges
        
        # Fallback to LUW
        return _seed_construct(k, g)
        
    else:
        return _seed_construct(k, g)
=======
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k == 3 and g == 16:
        # Base graph: Cube Q3
        # Vertices 0..7
        base_edges = []
        for u in range(8):
            for v in range(u + 1, 8):
                if bin(u ^ v).count('1') == 1:
                    base_edges.append((u, v))
        
        group_size = 117
        n_lifted = 8 * group_size
        
        # Precompute base adjacency
        base_adj = [[] for _ in range(8)]
        for u, v in base_edges:
            base_adj[u].append(v)
            base_adj[v].append(u)
        
        # Helper to check girth using non-backtracking walks
        def check_girth_fast(voltages, target_girth):
            """Check if the lifted graph has girth >= target_girth.
            Uses non-backtracking walk enumeration from each vertex.
            """
            # Build lifted adjacency as (base_vertex, offset) pairs
            # Vertex (u, x) connects to (v, x + voltages[(u,v)]) for each v in base_adj[u]
            
            def get_neighbors(u, x):
                """Get neighbors of lifted vertex (u, x)"""
                result = []
                for v in base_adj[u]:
                    key = (u, v) if u < v else (v, u)
                    if u < v:
                        offset = voltages[key]
                    else:
                        offset = (-voltages[key]) % group_size
                    y = (x + offset) % group_size
                    result.append((v, y))
                return result
            
            # BFS with non-backtracking constraint
            # State: (current_vertex, previous_vertex)
            # We track the shortest non-backtracking walk length
            
            max_depth = target_girth // 2
            
            for start_u in range(8):
                for start_x in range(group_size):
                    # BFS from (start_u, start_x)
                    # dist[(u, x, prev_u, prev_x)] = length
                    # But we can be smarter: for each (u, x), track the shortest
                    # non-backtracking walk from start that arrives at (u, x)
                    # without coming from any specific neighbor
                    
                    # Actually, let's use a simpler approach:
                    # For each starting vertex, do BFS but track parent to avoid backtracking
                    dist = {}
                    parent = {}
                    start_state = (start_u, start_x, -1, -1)
                    dist[start_state] = 0
                    q = deque([start_state])
                    
                    found_short_cycle = False
                    while q:
                        u, x, pu, px = q.popleft()
                        d = dist[(u, x, pu, px)]
                        if d >= max_depth:
                            continue
                        for v, y in get_neighbors(u, x):
                            # Non-backtracking: can't go back to (pu, px)
                            if v == pu and y == px:
                                continue
                            new_state = (v, y, u, x)
                            if new_state not in dist:
                                dist[new_state] = d + 1
                                parent[new_state] = (u, x, pu, px)
                                q.append(new_state)
                            else:
                                # We found a cycle
                                cycle_len = d + dist[new_state] + 1
                                if cycle_len < target_girth:
                                    return False
                    # Continue to next start vertex
                    
            return True

        # Random voltage search with optimization
        for attempt in range(20000):
            if time.time() > deadline:
                break
                
            voltages = {}
            for u, v in base_edges:
                vu = random.randint(0, group_size - 1)
                voltages[(u, v)] = vu
                voltages[(v, u)] = (-vu) % group_size
            
            if check_girth_fast(voltages, g):
                # Build the edge list
                lifted_edges = []
                for u, v in base_edges:
                    vu = voltages[(u, v)]
                    for x in range(group_size):
                        y = (x + vu) % group_size
                        u_lift = u * group_size + x
                        v_lift = v * group_size + y
                        lifted_edges.append((u_lift, v_lift))
                return n_lifted, lifted_edges
        
        # Fallback to LUW
        return _seed_construct(k, g)
        
    else:
        return _seed_construct(k, g)
>>>>>>> REPLACE

