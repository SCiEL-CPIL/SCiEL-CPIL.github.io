import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Precompute edge endpoints as arrays for speed
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    
    # Precompute lift adjacency indices for each edge
    # For edge i connecting (u,v):
    #   +1: (2u,2v), (2u+1,2v+1)
    #   -1: (2u,2v+1), (2u+1,2v)
    # We precompute the 4 vertex pairs for +1 and -1
    pos_pairs = np.array([(2*eu[i], 2*ev[i]) for i in range(num_edges)])
    neg_pairs = np.array([(2*eu[i], 2*ev[i]+1) for i in range(num_edges)])
    
    # For fast connectivity: precompute for each lift vertex, which edge indices touch it
    # and what the partner vertex would be for +1 and -1
    lift_n = 2 * n
    # adj_pos[v] = list of (edge_idx, partner_vertex) for +1
    # adj_neg[v] = list of (edge_idx, partner_vertex) for -1
    adj_pos = [[] for _ in range(lift_n)]
    adj_neg = [[] for _ in range(lift_n)]
    
    for i in range(num_edges):
        u, v = edges[i]
        # +1 edges: (u,0)-(v,0) and (u,1)-(v,1)
        a, b = 2*u, 2*v
        adj_pos[a].append((i, b))
        adj_pos[b].append((i, a))
        a, b = 2*u+1, 2*v+1
        adj_pos[a].append((i, b))
        adj_pos[b].append((i, a))
        # -1 edges: (u,0)-(v,1) and (u,1)-(v,0)
        a, b = 2*u, 2*v+1
        adj_neg[a].append((i, b))
        adj_neg[b].append((i, a))
        a, b = 2*u+1, 2*v
        adj_neg[a].append((i, b))
        adj_neg[b].append((i, a))
    
    def is_connected_lift(signs):
        """Fast connectivity check using bitset and precomputed adjacency."""
        # Use a simple BFS with a visited bitmask
        visited = 0
        visited |= 1  # start at vertex 0
        stack = [0]
        count = 1
        while stack:
            node = stack.pop()
            # Determine which edges incident to this node are active
            # We need to check both +1 and -1 adjacencies based on signs
            # For each neighbor in adj_pos[node], the edge is active if signs[edge_idx] == 1
            # For each neighbor in adj_neg[node], the edge is active if signs[edge_idx] == -1
            for (ei, partner) in adj_pos[node]:
                if signs[ei] == 1:
                    bit = 1 << partner
                    if not (visited & bit):
                        visited |= bit
                        stack.append(partner)
                        count += 1
            for (ei, partner) in adj_neg[node]:
                if signs[ei] == -1:
                    bit = 1 << partner
                    if not (visited & bit):
                        visited |= bit
                        stack.append(partner)
                        count += 1
        return count == lift_n
    
    def spectral_radius(signs):
        """Build signed adjacency matrix and compute spectral radius."""
        A = np.zeros((n, n))
        # Vectorized assignment
        A[eu, ev] = signs.astype(np.float64)
        A[ev, eu] = signs.astype(np.float64)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def evaluate(signs):
        """Returns (is_valid, radius)."""
        if not is_connected_lift(signs):
            return False, None
        return True, spectral_radius(signs)
    
    # Local search with early termination
    def local_search(initial_signs, max_iter=3000):
        signs = initial_signs.copy()
        valid, radius = evaluate(signs)
        if not valid:
            return None, None
        
        best_radius = radius
        best_signs = signs.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(num_edges):
                signs[i] = -signs[i]
                valid, r = evaluate(signs)
                if valid and r < best_radius:
                    best_radius = r
                    best_signs = signs.copy()
                    improved = True
                else:
                    signs[i] = -signs[i]  # revert
            if not improved:
                break
        
        return best_signs, best_radius
    
    # Multiple restarts
    rng = np.random.default_rng(3)
    best_overall = None
    best_overall_radius = float('inf')
    
    # Baseline random search
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges)
        valid, r = evaluate(signs)
        if valid and r < best_overall_radius:
            best_overall_radius = r
            best_overall = signs.copy()
    
    if best_overall is None:
        for _ in range(2000):
            signs = rng.choice([-1, 1], size=num_edges)
            valid, r = evaluate(signs)
            if valid:
                best_overall = signs.copy()
                best_overall_radius = r
                break
    
    if best_overall is None:
        return np.ones(num_edges, dtype=np.int8)
    
    # Local search from multiple starting points with more restarts
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            start = best_overall.copy()
        else:
            # Perturb best solution with varying amounts of noise
            if restart % 3 == 1:
                num_flips = 1
            elif restart % 3 == 2:
                num_flips = 2
            else:
                num_flips = 3
            start = best_overall.copy()
            flip_idx = rng.choice(num_edges, size=num_flips, replace=False)
            for i in flip_idx:
                start[i] = -start[i]
        
        result_signs, result_radius = local_search(start, max_iter=3000)
        
        if result_signs is not None and result_radius < best_overall_radius:
            best_overall_radius = result_radius
            best_overall = result_signs.copy()
    
    # Final polish: try a few more flips on the best solution
    if time.time() < deadline - 2.0:
        for _ in range(50):
            if time.time() > deadline:
                break
            signs = best_overall.copy()
            flip_idx = rng.choice(num_edges, size=1, replace=False)
            signs[flip_idx[0]] = -signs[flip_idx[0]]
            valid, r = evaluate(signs)
            if valid and r < best_overall_radius:
                best_overall_radius = r
                best_overall = signs.copy()
    
    return best_overall.astype(np.int8)
