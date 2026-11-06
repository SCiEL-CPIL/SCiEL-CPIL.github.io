"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _creates_c4(adj, u, v):
    """
    Check if adding edge (u, v) to the graph with adjacency sets 'adj' creates a C4.
    A C4 is u-x-y-v-u.
    This exists if there is an edge (x, y) such that x in adj[u] and y in adj[v].
    """
    nu = adj[u]
    nv = adj[v]
    # Optimize: iterate over the smaller set
    if len(nu) > len(nv):
        nu, nv = nv, nu
    for x in nu:
        if adj[x] & nv:
            return True
    return False

def _construct_c4_free(n, min_deg_required, time_limit):
    """
    Construct a C4-free graph on n vertices with min degree >= min_deg_required.
    Uses a deficit-focused greedy with repair and polish.
    """
    start_time = time.time()
    
    max_attempts = 300
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            return None
            
        adj = [set() for _ in range(n)]
        edges = set()
        
        # Phase 1: Deficit-focused greedy construction
        # Repeatedly pick the most deficient vertex and try to connect it
        max_steps = n * n * 4
        repair_steps_remaining = 150
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return None
                
            # Check success
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_required:
                break
                
            # Find the most deficient vertex (lowest degree, tie-break by random)
            # Use a small random perturbation to avoid deterministic behavior
            deficient_vertices = [i for i in range(n) if len(adj[i]) < min_deg_required]
            if not deficient_vertices:
                break
            # Pick the one with lowest degree, with random tie-breaking
            min_deg_val = min(len(adj[i]) for i in deficient_vertices)
            candidates_def = [i for i in deficient_vertices if len(adj[i]) == min_deg_val]
            u = random.choice(candidates_def)
            
            # Try to find a valid neighbor for u
            found = False
            non_neighbors = [v for v in range(n) if v != u and v not in adj[u]]
            random.shuffle(non_neighbors)
            
            for v in non_neighbors:
                if not _creates_c4(adj, u, v):
                    # Add edge
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_key = (min(u, v), max(u, v))
                    edges.add(edge_key)
                    found = True
                    break
                    
            if not found:
                # Stuck. Try to repair by removing an edge.
                if repair_steps_remaining <= 0:
                    break
                repair_steps_remaining -= 1
                
                if not edges:
                    break
                    
                # Prefer removing edges that don't drop any vertex below min_deg_required - 1
                # This keeps us close to the target
                removable = []
                for (a, b) in edges:
                    # Check if removing (a,b) keeps both degrees >= min_deg_required - 1
                    # Actually, we want to be able to re-add, so let's be more permissive
                    # Remove edges incident to high-degree vertices preferentially
                    removable.append((a, b))
                    
                random.shuffle(removable)
                u_rem, v_rem = removable[0]
                adj[u_rem].remove(v_rem)
                adj[v_rem].remove(u_rem)
                edges.remove((u_rem, v_rem))
                
        # Phase 2: Polish - try to add any remaining edges that don't create C4
        # This can help if we're close but not quite there
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if time.time() - start_time > time_limit:
                break
            if v in adj[u]:
                continue
            if not _creates_c4(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.add((min(u, v), max(u, v)))
                
        # Final check
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= min_deg_required:
            edge_list = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edge_list.append([u, v])
            return {'n': n, 'edges': edge_list}
            
    return None

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Try from largest n down
    # n=47 requires min degree 8.
    # n=46 requires min degree 7.
    
    time_budgets = {
        47: 35.0,
        46: 45.0,
        45: 20.0,
        44: 10.0,
        43: 5.0,
        42: 5.0,
        41: 5.0,
        40: 5.0,
        39: 2.0
    }
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
            
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2:
            break
            
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        # Cap the time for this n
        t_limit = min(time_budgets.get(n, 5.0), remaining_time - 1)
        
        if t_limit <= 0:
            continue
            
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result is not None:
            return result
            
    # Fallback: Construct a known small C4-free graph
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
