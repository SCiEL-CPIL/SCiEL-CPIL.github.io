"""Improved program for spectral graph lifts task."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    """
    Assign signs to edges of a prism graph to minimize the spectral radius
    of the signed adjacency matrix while ensuring the 2-lift is connected.
    
    Strategy: Use cycle balancing. For each cycle in the graph, the product
    of edge signs around the cycle determines the eigenvalues. By enforcing
    that the product around certain key cycles is -1, we can push eigenvalues
    toward zero.
    
    The prism graph has:
    - Two m-cycles (top: 0..m-1, bottom: m..2m-1)
    - Matching edges (i, i+m)
    
    Key cycles to balance:
    1. Top m-cycle
    2. Bottom m-cycle  
    3. Each "square" cycle (i, i+1, i+1+m, i+m) for i in 0..m-1
    4. The "long" cycles going around
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    m = graph['n'] // 2
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge index lookup
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    def get_sign_array(signs):
        """Convert signs dict to array."""
        arr = np.ones(num_edges, dtype=np.float64)
        for i in range(num_edges):
            arr[i] = 1.0 if signs.get(i, 1) == 1 else -1.0
        return arr
    
    def build_signed_adj(signs_arr):
        """Build signed adjacency matrix."""
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = signs_arr[i]
            A[u, v] += s
            A[v, u] += s
        return A
    
    def spectral_radius(signs_arr):
        """Compute spectral radius of signed adjacency matrix."""
        A = build_signed_adj(signs_arr)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_lift_connected(signs_arr):
        """Check if the 2-lift is connected.
        
        The 2-lift has vertices (v, 0) and (v, 1) for each vertex v.
        An edge (u, v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1).
        An edge (u, v) with sign -1 connects (u,0)-(v,1) and (u,1)-(v,0).
        
        The lift is connected iff there's no non-trivial partition of the
        2n vertices into two sets of n each that are disconnected.
        
        Equivalently, the lift is disconnected iff there exists a vector x 
        in {+1,-1}^n such that for every edge (u,v) with sign s, we have
        x[u]*s = x[v] (for s=+1) or x[u]*s = -x[v] (for s=-1)... 
        Actually, let me think more carefully.
        
        The lift graph has 2n vertices. It's disconnected iff there's a 
        bipartition. One way to check: do BFS/DFS on the lift graph.
        """
        # Build adjacency of the lift graph
        lift_adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = int(signs_arr[i])
            if s == 1:
                # +1: connects same sides
                lift_adj[u].append(v)
                lift_adj[v].append(u)
                lift_adj[n + u].append(n + v)
                lift_adj[n + v].append(n + u)
            else:
                # -1: connects opposite sides
                lift_adj[u].append(n + v)
                lift_adj[n + v].append(u)
                lift_adj[n + u].append(v)
                lift_adj[v].append(n + u)
        
        # BFS from vertex 0
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == 2 * n
    
    def evaluate(signs_dict):
        """Evaluate a sign assignment. Returns (radius, is_valid)."""
        signs_arr = get_sign_array(signs_dict)
        if not is_lift_connected(signs_arr):
            return (float('inf'), False)
        radius = spectral_radius(signs_arr)
        return (radius, True)
    
    # Identify the structure of the prism graph
    # Top cycle: 0 -> 1 -> 2 -> ... -> m-1 -> 0
    # Bottom cycle: m -> m+1 -> ... -> 2m-1 -> m
    # Matching: i <-> i+m
    
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    # Strategy 1: Analytical approach - set signs to balance cycles
    # For the lift to be connected, we need at least one negative sign 
    # in some cycle that creates "twisting"
    
    # The key insight: the signed adjacency matrix eigenvalues come in pairs
    # ±λ. The spectral radius is max |λ|.
    
    # For a balanced signed graph (product of signs around every cycle = +1),
    # the spectrum is the same as the unsigned graph.
    # For an unbalanced signed graph, eigenvalues can be smaller.
    
    # Let's try a systematic approach:
    # 1. Start with all +1 signs
    # 2. Try flipping individual edges and pairs to reduce spectral radius
    # 3. Use local search / simulated annealing
    
    best_signs = None
    best_radius = float('inf')
    
    # Initial candidate: all +1
    signs0 = {i: 1 for i in range(num_edges)}
    r0, valid0 = evaluate(signs0)
    if valid0 and r0 < best_radius:
        best_radius = r0
        best_signs = dict(signs0)
    
    # Candidate: make top cycle have product -1
    if top_cycle_edges:
        signs1 = {i: 1 for i in range(num_edges)}
        signs1[top_cycle_edges[0]] = -1  # Flip one edge in top cycle
        r1, valid1 = evaluate(signs1)
        if valid1 and r1 < best_radius:
            best_radius = r1
            best_signs = dict(signs1)
    
    # Candidate: make bottom cycle have product -1
    if bottom_cycle_edges:
        signs2 = {i: 1 for i in range(num_edges)}
        signs2[bottom_cycle_edges[0]] = -1
        r2, valid2 = evaluate(signs2)
        if valid2 and r2 < best_radius:
            best_radius = r2
            best_signs = dict(signs2)
    
    # Candidate: flip a matching edge
    if matching_edges:
        signs3 = {i: 1 for i in range(num_edges)}
        signs3[matching_edges[0]] = -1
        r3, valid3 = evaluate(signs3)
        if valid3 and r3 < best_radius:
            best_radius = r3
            best_signs = dict(signs3)
    
    # Local search: hill climbing with random restarts
    rng = np.random.default_rng(42)
    
    def random_signs():
        return {i: rng.choice([-1, 1]) for i in range(num_edges)}
    
    def flip_random(signs_dict, num_flips=1):
        new_signs = dict(signs_dict)
        for _ in range(num_flips):
            idx = rng.integers(0, num_edges)
            new_signs[idx] = -new_signs[idx]
        return new_signs
    
    # Hill climbing with restarts
    max_iterations = 5000
    iters_per_restart = 1000
    num_restarts = 5
    
    current_best = dict(best_signs) if best_signs else random_signs()
    current_radius, current_valid = evaluate(current_best)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart > 0:
            current_best = random_signs()
            current_radius, current_valid = evaluate(current_best)
            if not current_valid:
                continue
        
        for it in range(iters_per_restart):
            if time.time() > deadline:
                break
            
            # Try flipping 1 or 2 edges
            num_flips = 1 if rng.random() < 0.7 else 2
            candidate = flip_random(current_best, num_flips)
            cand_radius, cand_valid = evaluate(candidate)
            
            if cand_valid:
                if cand_radius < current_radius:
                    current_best = candidate
                    current_radius = cand_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = dict(current_best)
                elif rng.random() < 0.1:  # Accept with small probability for diversity
                    current_best = candidate
                    current_radius = cand_radius
            else:
                # Invalid, try to fix by flipping back
                pass
    
    # Also try targeted flips based on cycle structure
    if time.time() < deadline:
        # Try making specific cycles unbalanced
        target_configs = []
        
        # Config: top cycle product = -1, bottom cycle product = -1
        if top_cycle_edges and bottom_cycle_edges:
            cfg = {i: 1 for i in range(num_edges)}
            cfg[top_cycle_edges[0]] = -1
            cfg[bottom_cycle_edges[0]] = -1
            target_configs.append(cfg)
        
        # Config: alternating signs on matching
        if matching_edges:
            cfg = {i: 1 for i in range(num_edges)}
            for j in range(0, len(matching_edges), 2):
                cfg[matching_edges[j]] = -1
            target_configs.append(cfg)
        
        # Config: all matching edges negative
        if matching_edges:
            cfg = {i: 1 for i in range(num_edges)}
            for me in matching_edges:
                cfg[me] = -1
            target_configs.append(cfg)
        
        for cfg in target_configs:
            if time.time() > deadline:
                break
            r, valid = evaluate(cfg)
            if valid and r < best_radius:
                best_radius = r
                best_signs = dict(cfg)
    
    # Final polish: local search from best
    if best_signs is not None and time.time() < deadline:
        polish_start = time.time()
        polish_deadline = min(deadline, polish_start + 5.0)
        current = dict(best_signs)
        cur_r, cur_v = evaluate(current)
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(num_edges):
                if time.time() > polish_deadline:
                    break
                trial = dict(current)
                trial[i] = -trial[i]
                tr, tv = evaluate(trial)
                if tv and tr < cur_r:
                    current = trial
                    cur_r = tr
                    improved = True
                    if cur_r < best_radius:
                        best_radius = cur_r
                        best_signs = dict(current)
    
    # Return best signs as array
    if best_signs is None:
        # Fallback: all +1
        best_signs = {i: 1 for i in range(num_edges)}
    
    result = np.array([best_signs[i] for i in range(num_edges)], dtype=np.int8)
    return result

# EVOLVE-BLOCK-END
