"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 15
    # Total possible edges
    total_edges = n * (n - 1) // 2
    
    # Precompute edge map
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[idx] = (u, v)
            idx += 1
            
    # Reverse map: (u, v) -> index
    edge_index = {(v, u): i for i, (u, v) in edge_map.items()}
    edge_index.update({(u, v): i for i, (u, v) in edge_map.items()})

    # Structural Initialization:
    # Split vertices into A (0..6) and B (7..14)
    # Start with complete bipartite graph K_{7,8}
    # Degrees: A vertices have deg 8, B vertices have deg 7.
    # This creates a clear degree separation.
    
    A = list(range(7))
    B = list(range(7, 15))
    
    current_edges = set()
    for u in A:
        for v in B:
            current_edges.add(edge_index[(u, v)])
    
    # Incremental state tracking
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for e in current_edges:
        u, v = edge_map[e]
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    # Helper to check if adding edge idx is valid (incremental)
    def can_add_edge(edge_idx):
        u, v = edge_map[edge_idx]
        # Temporarily add the edge
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        
        # Check validity for all edges incident to u or v (since only their neighborhoods changed)
        # Actually, we need to check all edges because degrees of u and v changed,
        # which affects paths through any edge where u or v is an endpoint.
        # But we can be smart: only edges (a,b) where a is neighbor of u or b is neighbor of v
        # or a is neighbor of v or b is neighbor of u need rechecking.
        # Simpler: check all edges, but with early termination.
        
        valid = True
        for e in current_edges:
            if e == edge_idx:
                continue
            eu, ev = edge_map[e]
            # Candidates for s: neighbors of eu excluding ev
            # Candidates for t: neighbors of ev excluding eu
            s_by_deg = {}
            for s in adj[eu]:
                if s == ev: 
                    continue
                d = deg[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            t_by_deg = {}
            for t in adj[ev]:
                if t == eu: 
                    continue
                d = deg[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    if not (len(S_set) == 1 and len(T_set) == 1 and S_set == T_set):
                        valid = False
                        break
            if not valid:
                break
        
        # Rollback
        adj[u].remove(v)
        adj[v].remove(u)
        deg[u] -= 1
        deg[v] -= 1
        
        return valid
    
    # Helper to check full validity
    def is_valid(edge_set):
        adj_check = [set() for _ in range(n)]
        deg_check = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj_check[u].add(v)
            adj_check[v].add(u)
            deg_check[u] += 1
            deg_check[v] += 1
        
        for e in edge_set:
            u, v = edge_map[e]
            s_by_deg = {}
            for s in adj_check[u]:
                if s == v: 
                    continue
                d = deg_check[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            t_by_deg = {}
            for t in adj_check[v]:
                if t == u: 
                    continue
                d = deg_check[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    if not (len(S_set) == 1 and len(T_set) == 1 and S_set == T_set):
                        return False
                        
        return True

    # Verify initial state
    if not is_valid(current_edges):
        star_edges = set()
        for v in range(1, n):
            star_edges.add(edge_index[(0, v)])
        return [[u, v] for e in star_edges for u, v in [edge_map[e]]]

    best_edges = current_edges.copy()
    best_score = len(current_edges)
    
    # Generate list of missing internal edges
    missing_internal = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            u, v = A[i], A[j]
            idx = edge_index[(u, v)]
            if idx not in current_edges:
                missing_internal.append(idx)
                
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            u, v = B[i], B[j]
            idx = edge_index[(u, v)]
            if idx not in current_edges:
                missing_internal.append(idx)
                
    random.shuffle(missing_internal)
    
    # Greedy Add with incremental validity
    for idx in missing_internal:
        if time.time() > deadline:
            break
        if can_add_edge(idx):
            u, v = edge_map[idx]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            current_edges.add(idx)
            best_edges = current_edges.copy()
            best_score = len(current_edges)
            
    # Second pass: try all remaining missing edges
    all_missing = [i for i in range(total_edges) if i not in current_edges]
    random.shuffle(all_missing)
    
    for idx in all_missing:
        if time.time() > deadline:
            break
        if can_add_edge(idx):
            u, v = edge_map[idx]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            current_edges.add(idx)
            best_edges = current_edges.copy()
            best_score = len(current_edges)

    # Swap-based local search to escape local optima
    # Try: remove edge e1, add edge e2 (if valid and same count) or add e2, e3 after removing e1
    remaining_missing = [i for i in range(total_edges) if i not in current_edges]
    
    # Try single swaps: remove one, add one
    swap_attempts = 0
    while time.time() < deadline and swap_attempts < 500:
        swap_attempts += 1
        if not remaining_missing:
            break
        # Pick random missing edge to add
        add_idx = random.choice(remaining_missing)
        # Pick random existing edge to remove
        if not current_edges:
            break
        remove_idx = random.choice(list(current_edges))
        
        # Check if we can remove remove_idx and add add_idx
        # First remove remove_idx
        ru, rv = edge_map[remove_idx]
        adj[ru].remove(rv)
        adj[rv].remove(ru)
        deg[ru] -= 1
        deg[rv] -= 1
        current_edges.discard(remove_idx)
        
        # Try to add add_idx
        if can_add_edge(add_idx):
            u, v = edge_map[add_idx]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            current_edges.add(add_idx)
            best_edges = current_edges.copy()
            best_score = len(current_edges)
            remaining_missing.remove(add_idx)
        else:
            # Rollback: add back remove_idx
            adj[ru].add(rv)
            adj[rv].add(ru)
            deg[ru] += 1
            deg[rv] += 1
            current_edges.add(remove_idx)
    
    # Try double swaps: remove one, add two
    remaining_missing = [i for i in range(total_edges) if i not in current_edges]
    double_swap_attempts = 0
    while time.time() < deadline and double_swap_attempts < 200:
        double_swap_attempts += 1
        if len(remaining_missing) < 2:
            break
        # Pick two random missing edges to add
        add_idx1, add_idx2 = random.sample(remaining_missing, 2)
        # Pick random existing edge to remove
        if not current_edges:
            break
        remove_idx = random.choice(list(current_edges))
        
        # Remove remove_idx
        ru, rv = edge_map[remove_idx]
        adj[ru].remove(rv)
        adj[rv].remove(ru)
        deg[ru] -= 1
        deg[rv] -= 1
        current_edges.discard(remove_idx)
        
        # Try to add add_idx1
        if can_add_edge(add_idx1):
            u, v = edge_map[add_idx1]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            current_edges.add(add_idx1)
            
            # Try to add add_idx2
            if can_add_edge(add_idx2):
                u, v = edge_map[add_idx2]
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                current_edges.add(add_idx2)
                best_edges = current_edges.copy()
                best_score = len(current_edges)
                remaining_missing.remove(add_idx1)
                if add_idx2 in remaining_missing:
                    remaining_missing.remove(add_idx2)
            else:
                # Rollback add_idx1
                u, v = edge_map[add_idx1]
                adj[u].remove(v)
                adj[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
                current_edges.discard(add_idx1)
        # Rollback remove_idx
        adj[ru].add(rv)
        adj[rv].add(ru)
        deg[ru] += 1
        deg[rv] += 1
        current_edges.add(remove_idx)
    
    # Final validation
    if not is_valid(best_edges):
         star_edges = set()
         for v in range(1, n):
             star_edges.add(edge_index[(0, v)])
         best_edges = star_edges

    result = []
    for e in best_edges:
        u, v = edge_map[e]
        result.append([u, v])
        
    return result
# EVOLVE-BLOCK-END
