import time
import random

# EVOLVE-BLOCK-START
_N = 15
_ALL_EDGES = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin before 90s kill

def _norm(e):
    a, b = e
    return (a, b) if a < b else (b, a)

def _build_adj_deg(edges):
    """Build adjacency sets and degree array from a set of edges."""
    adj = [set() for _ in range(_N)]
    deg = [0] * _N
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _count_violations(adj, deg, edges):
    """Count the number of violating paths s-u-v-t where deg[s] == deg[t]."""
    count = 0
    for u, v in edges:
        # For each edge (u, v), check paths s-u-v-t and s-v-u-t
        # Path s-u-v-t: s is neighbor of u (s!=v), t is neighbor of v (t!=u, t!=s)
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    count += 1
    return count

def _is_valid(adj, deg, edges):
    """Check if the graph defined by adj, deg, edges is valid."""
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
            if deg[s] == deg[t]:
                return False
    return True

def _solve_once(rng):
    """
    Perform one run of the subtractive search.
    1. Start with complete graph.
    2. Greedily remove edges to minimize violations.
    3. Re-add edges that can be added without breaking validity.
    """
    g = set(_ALL_EDGES)
    adj, deg = _build_adj_deg(g)
    
    # Phase 1: Remove edges until valid
    # We iterate until no violations exist.
    # To avoid infinite loops or getting stuck, we limit iterations or check time.
    max_iter = 500 
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
            
        # Check if valid
        if _is_valid(adj, deg, g):
            break
            
        # Find one violation to guide removal
        # We need to find a specific violating path to identify candidate edges
        found_violation = False
        candidate_edges = []
        
        for u, v in g:
            for s in adj[u]:
                if s == v: continue
                for t in adj[v]:
                    if t == u or t == s: continue
                    if deg[s] == deg[t]:
                        # Violation found: path s-u-v-t
                        # Candidate edges to remove: (s,u), (u,v), (v,t)
                        candidate_edges = [
                            _norm((s, u)), 
                            _norm((u, v)), 
                            _norm((v, t))
                        ]
                        found_violation = True
                        break
                if found_violation:
                    break
            if found_violation:
                break
        
        if not found_violation:
            break
            
        # Evaluate which candidate edge removal results in the fewest total violations
        best_edge_to_remove = None
        best_violation_count = float('inf')
        
        # Shuffle candidates to break ties randomly
        rng.shuffle(candidate_edges)
        
        for e in candidate_edges:
            if e not in g:
                continue
                
            # Simulate removal
            new_g = g - {e}
            new_adj, new_deg = _build_adj_deg(new_g)
            
            # Count violations in the new graph
            v_count = _count_violations(new_adj, new_deg, new_g)
            
            if v_count < best_violation_count:
                best_violation_count = v_count
                best_edge_to_remove = e
                
        if best_edge_to_remove is None:
            # Fallback: remove first candidate
            best_edge_to_remove = candidate_edges[0]
            
        # Perform removal
        e_rm = best_edge_to_remove
        g.discard(e_rm)
        adj, deg = _build_adj_deg(g)
        
        # If we still have violations, loop continues.
        # If we removed an edge but violations didn't decrease, we might oscillate.
        # The "minimize total violations" heuristic usually prevents this.
        
    # Phase 2: Re-add edges
    # Try to add back any edge from _ALL_EDGES not in g
    for e in _ALL_EDGES:
        if time.time() > _DEADLINE:
            break
        if e not in g:
            new_g = g | {e}
            new_adj, new_deg = _build_adj_deg(new_g)
            if _is_valid(new_adj, new_deg, new_g):
                g = new_g
                adj, deg = new_adj, new_deg
                
    return g

def construct():
    start_time = time.time()
    deadline = 85.0 # Hard limit inside construct for safety
    
    best_g = None
    best_size = -1
    
    # Run multiple restarts with different seeds
    base_seed = 12345
    restart_count = 0
    
    while time.time() - start_time < deadline:
        rng = random.Random(base_seed + 1000 * restart_count)
        
        g = _solve_once(rng)
        
        # Verify final validity (paranoia check)
        if g:
            adj, deg = _build_adj_deg(g)
            if _is_valid(adj, deg, g):
                if len(g) > best_size:
                    best_size = len(g)
                    best_g = g
                    # Early exit if we hit theoretical max or near max?
                    # Max edges is 105. If we get close, we can stop.
                    # But let's keep searching for the best.
        
        restart_count += 1
        
        # If we've done a reasonable number of restarts, break to ensure return
        if restart_count > 50:
            break

    if best_g is None:
        # Fallback: Star graph is always valid
        # Star on 15 vertices: edges (0,1)..(0,14)
        # Any path of length 3 s-u-v-t requires s!=t.
        # In a star, u and v must be 0 and some leaf? No, path is simple.
        # s-u-v-t.
        # If u=0, v=leaf1. s is leaf2 (neighbor of 0). t must be neighbor of leaf1.
        # Leaf1 only connects to 0. So t must be 0. But t!=u (0) is required? 
        # Path s-u-v-t is simple, so s,u,v,t distinct.
        # If v=leaf1, adj[v]={0}. t must be 0. But u=0. So t=u, not simple.
        # Thus no simple path of length 3 exists in a star.
        # So star is valid.
        best_g = {(0, i) for i in range(1, _N)}
        
    # Convert to required format: list of [u, v] with Python ints
    result = []
    for u, v in best_g:
        result.append([int(u), int(v)])
        
    return result
# EVOLVE-BLOCK-END
