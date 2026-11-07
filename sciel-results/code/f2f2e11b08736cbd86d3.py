import random
import time
from itertools import combinations, permutations

def construct():
    n = 15
    
    # Precompute all simple paths of length 3 (4 vertices s, u, v, t)
    # where s, u, v, t are all distinct
    paths = []
    vertices = list(range(n))
    for s in vertices:
        for u in vertices:
            if u == s: continue
            for v in vertices:
                if v == s or v == u: continue
                for t in vertices:
                    if t == s or t == u or t == v: continue
                    paths.append((s, u, v, t))
    
    num_paths = len(paths)
    
    # Precompute for each path the three edge indices (as (min,max) pairs)
    # and the endpoint pair (s, t)
    path_edges = []  # list of (e1, e2, e3, s, t) where e's are (min, max) tuples
    for s, u, v, t in paths:
        e1 = (min(s, u), max(s, u))
        e2 = (min(u, v), max(u, v))
        e3 = (min(v, t), max(v, t))
        path_edges.append((e1, e2, e3, s, t))
    
    def is_valid_fast(adj_matrix, deg):
        """Check validity using adjacency matrix for O(1) edge lookups."""
        for e1, e2, e3, s, t in path_edges:
            if adj_matrix[e1[0]][e1[1]] and adj_matrix[e2[0]][e2[1]] and adj_matrix[e3[0]][e3[1]]:
                if deg[s] == deg[t]:
                    return False
        return True
    
    def get_adj_matrix(edge_set):
        """Convert edge set to adjacency matrix (list of bytearrays for speed)."""
        adj = [bytearray(n) for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj
    
    def get_degree(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def normalize_edge(u, v):
        if u > v:
            return (v, u)
        return (u, v)
    
    # All possible edges
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Local Search
    best_edges = set()
    best_count = 0
    
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # Initialize with K_{7,8}
    init_edges = set()
    for u in range(7):
        for v in range(7, 15):
            init_edges.add(normalize_edge(u, v))
    
    # Check init
    deg_init = get_degree(init_edges)
    adj_init = get_adj_matrix(init_edges)
    if is_valid_fast(adj_init, deg_init):
        best_edges = init_edges
        best_count = len(init_edges)
    
    # Local Search Loop
    while time.time() < deadline:
        current_edges = set(best_edges)
        current_deg = get_degree(current_edges)
        current_adj = get_adj_matrix(current_edges)
        
        # 1. Try to add edges greedily
        non_edges = list(all_edges - current_edges)
        random.shuffle(non_edges)
        
        added_any = False
        for e in non_edges:
            if time.time() > deadline:
                break
            u, v = e
            # Check if adding this edge maintains validity
            new_deg = current_deg[:]
            new_deg[u] += 1
            new_deg[v] += 1
            
            # Update adjacency
            current_adj[u][v] = 1
            current_adj[v][u] = 1
            
            if is_valid_fast(current_adj, new_deg):
                current_edges.add(e)
                current_deg = new_deg
                added_any = True
            else:
                # Revert
                current_adj[u][v] = 0
                current_adj[v][u] = 0
        
        if len(current_edges) > best_count:
            best_edges = current_edges
            best_count = len(current_edges)
        
        # 2. Try 1-swap moves (remove one, add one)
        if len(current_edges) > 0:
            edge_list = list(current_edges)
            random.shuffle(edge_list)
            
            improved = False
            for e_rem in edge_list:
                if time.time() > deadline:
                    break
                
                u_rem, v_rem = e_rem
                # Remove e_rem
                current_adj[u_rem][v_rem] = 0
                current_adj[v_rem][u_rem] = 0
                temp_deg = current_deg[:]
                temp_deg[u_rem] -= 1
                temp_deg[v_rem] -= 1
                
                # Try adding random non-edges (excluding e_rem)
                candidates = [c for c in non_edges if c != e_rem]
                random.shuffle(candidates)
                
                for e_add in candidates[:15]:
                    if time.time() > deadline:
                        break
                    u_add, v_add = e_add
                    if u_add == v_add:
                        continue
                    if (u_add, v_add) in current_edges or (v_add, u_add) in current_edges:
                        continue
                    
                    new_deg = temp_deg[:]
                    new_deg[u_add] += 1
                    new_deg[v_add] += 1
                    
                    current_adj[u_add][v_add] = 1
                    current_adj[v_add][u_add] = 1
                    
                    if is_valid_fast(current_adj, new_deg):
                        # Valid swap found
                        current_edges.discard(e_rem)
                        current_edges.add(e_add)
                        current_deg = new_deg
                        improved = True
                        break
                    else:
                        # Revert
                        current_adj[u_add][v_add] = 0
                        current_adj[v_add][u_add] = 0
                
                if improved:
                    if len(current_edges) > best_count:
                        best_edges = current_edges
                        best_count = len(current_edges)
                    break
                else:
                    # Revert removal
                    current_adj[u_rem][v_rem] = 1
                    current_adj[v_rem][u_rem] = 1
            
            if improved:
                # Restart from new state
                continue
        
        # 3. Try 2-swap moves (remove two, add two) - helps escape local minima
        if len(current_edges) >= 2 and time.time() < deadline:
            edge_list = list(current_edges)
            random.shuffle(edge_list)
            
            improved = False
            for i in range(min(20, len(edge_list))):
                if time.time() > deadline:
                    break
                e_rem1 = edge_list[i]
                for j in range(i + 1, min(i + 5, len(edge_list))):
                    if time.time() > deadline:
                        break
                    e_rem2 = edge_list[j]
                    
                    u1, v1 = e_rem1
                    u2, v2 = e_rem2
                    
                    # Remove both edges
                    current_adj[u1][v1] = 0
                    current_adj[v1][u1] = 0
                    current_adj[u2][v2] = 0
                    current_adj[v2][u2] = 0
                    temp_deg = current_deg[:]
                    temp_deg[u1] -= 1
                    temp_deg[v1] -= 1
                    temp_deg[u2] -= 1
                    temp_deg[v2] -= 1
                    
                    # Try adding two random non-edges
                    candidates = [c for c in non_edges if c != e_rem1 and c != e_rem2]
                    random.shuffle(candidates)
                    
                    for e_add1 in candidates[:8]:
                        if time.time() > deadline:
                            break
                        u3, v3 = e_add1
                        if (u3, v3) in current_edges or (v3, u3) in current_edges:
                            continue
                        
                        for e_add2 in candidates[:8]:
                            if time.time() > deadline:
                                break
                            if e_add2 == e_add1:
                                continue
                            u4, v4 = e_add2
                            if (u4, v4) in current_edges or (v4, u4) in current_edges:
                                continue
                            if (u3, v3) == (u4, v4) or (u3, v3) == (v4, u4):
                                continue
                            
                            new_deg = temp_deg[:]
                            new_deg[u3] += 1
                            new_deg[v3] += 1
                            new_deg[u4] += 1
                            new_deg[v4] += 1
                            
                            current_adj[u3][v3] = 1
                            current_adj[v3][u3] = 1
                            current_adj[u4][v4] = 1
                            current_adj[v4][u4] = 1
                            
                            if is_valid_fast(current_adj, new_deg):
                                # Valid 2-swap found
                                current_edges.discard(e_rem1)
                                current_edges.discard(e_rem2)
                                current_edges.add(e_add1)
                                current_edges.add(e_add2)
                                current_deg = new_deg
                                improved = True
                                break
                            else:
                                # Revert
                                current_adj[u3][v3] = 0
                                current_adj[v3][u3] = 0
                                current_adj[u4][v4] = 0
                                current_adj[v4][u4] = 0
                        
                        if improved:
                            break
                    
                    if improved:
                        if len(current_edges) > best_count:
                            best_edges = current_edges
                            best_count = len(current_edges)
                        break
                    else:
                        # Revert removals
                        current_adj[u1][v1] = 1
                        current_adj[v1][u1] = 1
                        current_adj[u2][v2] = 1
                        current_adj[v2][u2] = 1
                
                if improved:
                    break
            
            if improved:
                continue
        
        # Random restart with small probability
        if random.random() < 0.03 and time.time() < deadline:
            current_edges = set()
            for e in all_edges:
                if random.random() < 0.4:
                    current_edges.add(e)
            current_deg = get_degree(current_edges)
            current_adj = get_adj_matrix(current_edges)
            
            # Fix by removing problematic edges
            while not is_valid_fast(current_adj, current_deg) and current_edges:
                e_rem = random.choice(list(current_edges))
                current_edges.remove(e_rem)
                u, v = e_rem
                current_deg[u] -= 1
                current_deg[v] -= 1
                current_adj[u][v] = 0
                current_adj[v][u] = 1 if current_adj[v][u] else 0  # keep correct
                # Actually need to recompute properly
                current_adj[u][v] = 0
                current_adj[v][u] = 0
            
            if len(current_edges) > best_count:
                best_edges = current_edges
                best_count = len(current_edges)
    
    # Convert best_edges to list of lists
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result

if __name__ == "__main__":
    pass
