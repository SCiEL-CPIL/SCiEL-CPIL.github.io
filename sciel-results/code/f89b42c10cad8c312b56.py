import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # Safety margin
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Precompute edge map
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[idx] = (u, v)
            idx += 1
            
    # Reverse map: (u, v) -> index
    edge_index = {}
    for i, (u, v) in edge_map.items():
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    def is_valid(edge_set):
        """
        Fast validity check.
        Returns True if no path s-u-v-t (s!=t) exists with deg(s)==deg(t).
        """
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Check all edges (u, v)
        for e in edge_set:
            u, v = edge_map[e]
            
            # Group neighbors of u (excluding v) by degree
            s_by_deg = {}
            for s in adj[u]:
                if s == v:
                    continue
                d = deg[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            # Group neighbors of v (excluding u) by degree
            t_by_deg = {}
            for t in adj[v]:
                if t == u:
                    continue
                d = deg[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            # Check for conflicts
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    # A violation exists if there are s in S_set, t in T_set with s != t
                    # This fails only if S_set == T_set == {x} for some single x
                    if not (len(S_set) == 1 and len(T_set) == 1 and S_set == T_set):
                        return False
                        
        return True

    def find_violation(edge_set):
        """
        Find one violating path (s, u, v, t) or return None.
        """
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        for e in edge_set:
            u, v = edge_map[e]
            s_by_deg = {}
            for s in adj[u]:
                if s == v:
                    continue
                d = deg[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            t_by_deg = {}
            for t in adj[v]:
                if t == u:
                    continue
                d = deg[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    for s in S_set:
                        for t in T_set:
                            if s != t:
                                return (s, u, v, t)
        return None

    # Strategy:
    # 1. Structural Initialization: K_{7,8} bipartite graph.
    A = list(range(7))
    B = list(range(7, 15))
    
    best_edges_set = set()
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        random.seed(restart * 12345)
        
        current_edges = set()
        for u in A:
            for v in B:
                current_edges.add(edge_index[(u, v)])
                
        if not is_valid(current_edges):
            for v in range(1, n):
                current_edges.add(edge_index[(0, v)])
            if not is_valid(current_edges):
                continue
        
        best_in_restart = current_edges.copy()
        
        # Greedy Add Phase with degree-aware ordering
        missing = [i for i in range(total_edges) if i not in current_edges]
        
        # Sort missing edges by sum of endpoint degrees (lower first - safer to add)
        def edge_priority(idx):
            u, v = edge_map[idx]
            # Estimate degree: count edges incident to u and v in current_edges
            du = sum(1 for e in current_edges if edge_map[e][0] == u or edge_map[e][1] == u)
            dv = sum(1 for e in current_edges if edge_map[e][0] == v or edge_map[e][1] == v)
            return du + dv + random.random()  # Add small random tiebreaker
        
        missing.sort(key=edge_priority)
        
        for idx in missing:
            if time.time() > deadline:
                break
            
            new_set = current_edges | {idx}
            if is_valid(new_set):
                current_edges = new_set
                if len(current_edges) > len(best_in_restart):
                    best_in_restart = current_edges.copy()
        
        # Local Search / Refinement - multiple passes
        remaining_missing = [i for i in range(total_edges) if i not in current_edges]
        
        for _ in range(5):
            if time.time() > deadline:
                break
            if not remaining_missing:
                break
            
            random.shuffle(remaining_missing)
            progress = False
            for idx in remaining_missing[:]:
                if time.time() > deadline:
                    break
                
                new_set = current_edges | {idx}
                if is_valid(new_set):
                    current_edges = new_set
                    remaining_missing.remove(idx)
                    if len(current_edges) > len(best_in_restart):
                        best_in_restart = current_edges.copy()
                    progress = True
            
            if not progress:
                break
        
        if len(best_in_restart) > len(best_edges_set):
            best_edges_set = best_in_restart.copy()

    # Final check
    if not is_valid(best_edges_set):
        best_edges_set = set()
        for v in range(1, n):
            best_edges_set.add(edge_index[(0, v)])
        if not is_valid(best_edges_set):
            best_edges_set = set()

    # Convert to list of lists
    result = []
    for e in best_edges_set:
        u, v = edge_map[e]
        result.append([int(u), int(v)])
        
    return result
