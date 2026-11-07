"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
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
            
    # Helper to calculate validity
    def is_valid(edge_set):
        # Build adjacency and degrees
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Check all edges (u, v) for violating paths s-u-v-t
        for e in edge_set:
            u, v = edge_map[e]
            # Candidates for s: neighbors of u excluding v
            # Candidates for t: neighbors of v excluding u
            # If deg[s] == deg[t] and s != t, violation.
            
            s_by_deg = {}
            for s in adj[u]:
                if s == v: continue
                d = deg[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            t_by_deg = {}
            for t in adj[v]:
                if t == u: continue
                d = deg[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            # Check intersection of degrees
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    # We need s in S_set, t in T_set such that s != t.
                    # This fails ONLY if S_set == T_set == {x} for some x.
                    if not (len(S_set) == 1 and len(T_set) == 1 and S_set == T_set):
                        return False
                        
        return True

    # Verify initial state
    if not is_valid(current_edges):
        # Fallback to star if structural init fails (unlikely)
        star_edges = set()
        for v in range(1, n):
            star_edges.add(edge_index[(0, v)])
        return [[u, v] for e in star_edges for u, v in [edge_map[e]]]

    best_edges = current_edges.copy()
    best_score = len(current_edges)
    
    # Local Search:
    # We want to add edges.
    # Priority: Add internal edges (within A or within B).
    # Why? Adding cross edges (A-B) is already done.
    # Adding internal edges increases degrees within a partition.
    # This might cause violations if two vertices in the same partition
    # get connected to a common neighbor in the other partition with same degree endpoints.
    # But the initial degree gap (8 vs 7) provides buffer.
    
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
    
    # Greedy Add
    for idx in missing_internal:
        if time.time() > deadline:
            break
        new_edges = current_edges | {idx}
        if is_valid(new_edges):
            current_edges = new_edges
            best_edges = current_edges.copy()
            best_score = len(current_edges)
            
    # Now try to add remaining missing edges (including any cross edges if we removed any, 
    # but we didn't, so just internal ones that failed)
    # Actually, we can also try to add edges that were skipped.
    # Let's do a second pass with random order for any missing edges.
    
    all_missing = [i for i in range(total_edges) if i not in current_edges]
    random.shuffle(all_missing)
    
    for idx in all_missing:
        if time.time() > deadline:
            break
        new_edges = current_edges | {idx}
        if is_valid(new_edges):
            current_edges = new_edges
            best_edges = current_edges.copy()
            best_score = len(current_edges)

    # Simulated Annealing / Random Perturbation for further improvement
    # Try to add edges, and if stuck, try removing one to allow two others?
    # Given the density, greedy add usually finds a local optimum.
    # Let's try a few random restarts of the "greedy add" on slightly different seeds?
    # No, the structure is fixed. Let's just try random additions from the remaining pool.
    
    remaining_missing = [i for i in range(total_edges) if i not in current_edges]
    
    attempts = 0
    while time.time() < deadline and remaining_missing:
        attempts += 1
        if attempts > 1000: break
        
        # Pick a random missing edge
        idx = random.choice(remaining_missing)
        new_edges = current_edges | {idx}
        if is_valid(new_edges):
            current_edges = new_edges
            best_edges = current_edges.copy()
            best_score = len(current_edges)
            remaining_missing.remove(idx)
        else:
            # If adding fails, maybe we can swap?
            # Remove a random existing edge and try adding the target edge?
            # This is risky and might lower score. 
            # Given the structural advantage, simple addition is likely sufficient.
            # Let's just break if we can't add anything easily to save time?
            # No, keep trying random ones.
            pass
            
    # Final validation
    if not is_valid(best_edges):
         # Fallback
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
