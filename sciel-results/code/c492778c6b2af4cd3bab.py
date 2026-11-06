import time
import random

def construct():
    deadline = time.time() + 110
    n = 16
    
    def build_adj(edge_set):
        adj = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def is_triangle_free(edge_set, adj):
        for (u, v) in edge_set:
            if adj[u] & adj[v]:
                return False
        return True
    
    def is_3_colorable(adj):
        """Fast 3-colorability check for n=16"""
        # Backtracking with MRV
        uncolored = (1 << n) - 1
        color_of = [-1] * n
        
        def backtrack():
            if time.time() > deadline - 3:
                return None  # timeout: assume colorable
            
            # Find uncolored vertex with most colored neighbors (MRV)
            best_v = -1
            best_choices = 4  # > 3 means no choice
            for v in range(n):
                if color_of[v] != -1:
                    continue
                # Count available colors
                used = set()
                nb = adj[v]
                m = nb
                while m:
                    u = (m & -m).bit_length() - 1
                    m &= m - 1
                    if color_of[u] != -1:
                        used.add(color_of[u])
                choices = 3 - len(used)
                if choices < best_choices:
                    best_choices = choices
                    best_v = v
                    if choices == 0:
                        return False
                    if choices == 1:
                        break
            
            if best_v == -1:
                return True  # All colored
            
            v = best_v
            used = set()
            nb = adj[v]
            m = nb
            while m:
                u = (m & -m).bit_length() - 1
                m &= m - 1
                if color_of[u] != -1:
                    used.add(color_of[u])
            
            for c in range(3):
                if c not in used:
                    color_of[v] = c
                    if backtrack():
                        return True
                    color_of[v] = -1
            return False
        
        result = backtrack()
        return result
    
    # Grötzsch graph on vertices 0-10
    def groetzch_edges():
        edges = set()
        # C5: 0-1-2-3-4-0
        for i in range(5):
            edges.add((i, (i+1) % 5))
        # Mycielski layer: 5,6,7,8,9
        for i in range(5):
            edges.add((i, 5+i))
            edges.add((5+i, (i+1) % 5))
            edges.add((5+i, (i-1) % 5))
        # Top vertex 10
        for i in range(5):
            edges.add((10, 5+i))
        return edges
    
    best_edges = groetzch_edges()
    best_count = len(best_edges)
    
    def try_local_search(edge_set, time_limit):
        current = set(edge_set)
        adj = build_adj(current)
        
        # Precompute which edges can be added without creating triangles
        addable = []
        for u in range(n):
            for v in range(u+1, n):
                if (u,v) not in current and (v,u) not in current:
                    if (adj[u] & adj[v]) == 0:
                        addable.append((u,v))
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(addable)
            for (u,v) in addable:
                if (u,v) in current:
                    continue
                new_set = set(current)
                new_set.add((u,v))
                new_adj = build_adj(new_set)
                if not is_3_colorable(new_adj):
                    current = new_set
                    adj = new_adj
                    improved = True
                    # Rebuild addable
                    addable = []
                    for a in range(n):
                        for b in range(a+1, n):
                            if (a,b) not in current and (b,a) not in current:
                                if (adj[a] & adj[b]) == 0:
                                    addable.append((a,b))
                    break
        
        return current, len(current)
    
    # First, try to extend Grötzsch by connecting vertices 11-15
    current, current_count = try_local_search(best_edges, deadline - 30)
    if current_count > best_count:
        best_edges = current
        best_count = current_count
    
    # Try random restarts
    for _ in range(100):
        if time.time() > deadline - 20:
            break
        
        # Random triangle-free graph
        edges = set()
        all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
        random.shuffle(all_pairs)
        adj = [0] * n
        for (u,v) in all_pairs:
            if (adj[u] & adj[v]) == 0:
                edges.add((u,v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Check if non-3-colorable
        if not is_3_colorable(adj):
            if len(edges) > best_count:
                best_edges = edges
                best_count = len(edges)
        
        # Local search
        if time.time() < deadline - 15:
            current, current_count = try_local_search(edges, deadline - 15)
            if current_count > best_count:
                best_edges = current
                best_count = current_count
    
    return [[u, v] for (u, v) in best_edges]
