"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def find_hc(adj_cur, deadline):
        """
        Find a Hamiltonian cycle in the graph defined by adj_cur.
        Returns list of vertices or None.
        """
        # Check connectivity and degree constraints first for speed
        for i in range(n):
            if len(adj_cur[i]) < 2:
                return None
        
        path = [0]
        visited = {0}
        
        def dfs(u, depth):
            if time.time() > deadline - 2:
                return None
            
            if depth == n:
                # Check if we can close the cycle
                if 0 in adj_cur[u]:
                    return list(path)
                return None
            
            # Heuristic: Warnsdorff's rule - visit neighbors with fewest unvisited neighbors first
            neighbors = adj_cur[u]
            scored = []
            for v in neighbors:
                if v not in visited:
                    # Count unvisited neighbors of v
                    unvisited_count = 0
                    for w in adj_cur[v]:
                        if w not in visited:
                            unvisited_count += 1
                    scored.append((unvisited_count, v))
            
            # Shuffle to break ties and add randomness
            random.shuffle(scored)
            scored.sort(key=lambda x: x[0])
            
            for _, v in scored:
                path.append(v)
                visited.add(v)
                
                res = dfs(v, depth + 1)
                if res is not None:
                    return res
                
                path.pop()
                visited.remove(v)
            
            return None
        
        return dfs(0, 1)
    
    best_cycles = []
    
    # We want to find 3 edge-disjoint Hamiltonian cycles.
    # Since the graph is 6-regular, removing one HC leaves a 4-regular graph.
    # Removing another leaves a 2-regular graph (which is a disjoint union of cycles).
    # If the remaining 2-regular graph is a single cycle of length 21, we have 3 HCs.
    
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
            
        current_used = set()
        current_cycles = []
        
        success = True
        for k in range(3):
            # Build adjacency for available edges
            adj_cur = defaultdict(list)
            for u, v in edges:
                e = (min(u, v), max(u, v))
                if e not in current_used:
                    adj_cur[u].append(v)
                    adj_cur[v].append(u)
            
            # Quick check: if any vertex has degree < 2, we can't form a HC
            possible = True
            for i in range(n):
                if len(adj_cur[i]) < 2:
                    possible = False
                    break
            
            if not possible:
                success = False
                break
                
            hc = find_hc(adj_cur, deadline)
            
            if hc is None:
                success = False
                break
            
            current_cycles.append(hc)
            
            # Mark edges as used
            for i in range(len(hc)):
                u = hc[i]
                v = hc[(i + 1) % len(hc)]
                e = (min(u, v), max(u, v))
                current_used.add(e)
        
        if success and len(current_cycles) == 3:
            # Verify edge disjointness
            all_edges = set()
            valid = True
            for c in current_cycles:
                for i in range(len(c)):
                    u = c[i]
                    v = c[(i + 1) % len(c)]
                    e = (min(u, v), max(u, v))
                    if e in all_edges:
                        valid = False
                        break
                    all_edges.add(e)
            
            if valid:
                return [list(c) for c in current_cycles]
        
        # If we didn't get 3, keep the best partial result
        if len(current_cycles) > len(best_cycles):
            best_cycles = current_cycles
            
        # If we found 2, it's unlikely we'll find 3 with this randomization, but let's keep trying.
        # However, if we found 0, it might be due to bad luck.
        
    # Return best found (could be 0, 1, or 2)
    # If 0, we must return at least 1 valid cycle to be non-empty?
    # "construct() -> nonempty list of simple cycles"
    # If we fail to find any, we should probably find one greedily.
    
    if not best_cycles:
        # Fallback: find any single cycle
        # Just find a triangle or a long cycle
        # For a 6-regular graph, a triangle is very likely.
        for u, v in edges:
            common = set(adj[u]) & set(adj[v])
            for w in common:
                if w != u and w != v:
                    # Check edges (u,v), (v,w), (w,u)
                    if (min(u,v), max(u,v)) in edge_set and (min(v,w), max(v,w)) in edge_set and (min(u,w), max(u,w)) in edge_set:
                        return [[u, v, w]]
    
    # If still empty (shouldn't happen), return a dummy valid cycle? 
    # But it must be from the graph.
    # Let's try a simple DFS for any cycle.
    # Actually, if best_cycles is empty, it means we couldn't find even 1 HC in 500 tries?
    # That's extremely unlikely for this graph.
    # But to be safe, let's just return an empty list? No, "nonempty".
    # Let's force a single HC search without the "3 cycle" constraint.
    
    adj_full = defaultdict(list)
    for u, v in edges:
        adj_full[u].append(v)
        adj_full[v].append(u)
        
    hc = find_hc(adj_full, deadline)
    if hc:
        return [hc]
        
    # Absolute fallback: return first 3 vertices if they form a cycle?
    # This is risky. But the graph is dense.
    # Let's assume the loop above found at least 1.
    return best_cycles

# EVOLVE-BLOCK-END
