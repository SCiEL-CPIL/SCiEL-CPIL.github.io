"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n = graph['n']
    
    # Precompute edge arrays
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    # Identify edge types: cycle edges (within 0..m-1 or m..2m-1) vs matching edges (i, i+m)
    # Matching edges: u < m and v == u + m (or vice versa)
    # Cycle edges: both in same cycle
    
    matching_idx = []
    cycle_top_idx = []  # edges in 0..m-1
    cycle_bot_idx = []  # edges in m..2m-1
    
    for i, (u, v) in enumerate(edges):
        if u < m and v == u + m:
            matching_idx.append(i)
        elif v < m and u == v + m:
            matching_idx.append(i)
        elif u < m and v < m:
            cycle_top_idx.append(i)
        elif u >= m and v >= m:
            cycle_bot_idx.append(i)
    
    def build_matrix(signs):
        A = np.zeros((n, n))
        for i in range(n_edges):
            s = signs[i]
            A[u_arr[i], v_arr[i]] = s
            A[v_arr[i], u_arr[i]] = s
        return A
    
    def spectral_radius(signs):
        A = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        try:
            radius, details = check_lift(graph, signs)
            return True
        except Exception:
            return False
    
    def check_valid_fast(signs):
        """Fast validity check using the lift structure."""
        # The lift has 2n vertices. Each vertex (v, s) connects to (u, s*sign) for each edge (u,v) with sign.
        # For the lift to be 3-regular, each vertex must have degree 3 in the lift.
        # Since the base graph is 3-regular (prism graph), each vertex in base has degree 3.
        # In the lift, each vertex (v, s) connects to 3 neighbors, so it's automatically 3-regular.
        # The only thing to check is connectivity of the lift.
        
        # Build adjacency for the lift
        n_lift = 2 * n
        adj = [[] for _ in range(n_lift)]
        for i in range(n_edges):
            s = signs[i]
            u, v = edges[i]
            # Edge (u,v) with sign s: connects (u,0)-(v,s) and (u,1)-(v,1-s) in lift
            # More precisely: if sign is +1, (u,a) connects to (v,a) for a in {0,1}
            # If sign is -1, (u,a) connects to (v,1-a) for a in {0,1}
            for a in range(2):
                b = a if s == 1 else 1 - a
                adj[u * 2 + a].append(v * 2 + b)
                adj[v * 2 + b].append(u * 2 + a)
        
        # BFS for connectivity
        visited = [False] * n_lift
        visited[0] = True
        stack = [0]
        count = 1
        while stack:
            node = stack.pop()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    stack.append(nb)
        return count == n_lift
    
    # Initialize with random signs
    rng = np.random.default_rng(3)
    best_signs = None
    best_radius = float('inf')
    
    # First, try the seed's approach: sample random sign vectors
    for _ in range(1000):
        signs = rng.choice([-1, 1], size=n_edges)
        if check_valid_fast(signs):
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    if best_signs is None:
        # If no valid lift found, try all +1
        signs = np.ones(n_edges, dtype=int)
        if check_valid_fast(signs):
            best_signs = signs.copy()
            best_radius = spectral_radius(signs)
        else:
            # This shouldn't happen for prism graphs, but handle gracefully
            best_signs = rng.choice([-1, 1], size=n_edges)
            best_radius = spectral_radius(best_signs)
    
    # Local search with 2-opt moves
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Precompute which pairs of edges to try for 2-opt
    # Focus on pairs that are likely to have impact: matching edges with cycle edges
    all_indices = list(range(n_edges))
    
    improvement_threshold = 1e-10
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Try 2-opt moves: flip two edges at a time
        # To be efficient, sample a subset of pairs
        if time.time() > deadline - 5.0:
            break
            
        # Try all pairs involving matching edges with cycle edges
        # and also random pairs
        pairs_to_try = []
        
        # Matching-matching pairs
        if len(matching_idx) >= 2:
            for i, j in combinations(matching_idx, 2):
                pairs_to_try.append((i, j))
        
        # Matching-cycle pairs
        for mi in matching_idx:
            for ci in cycle_top_idx[:5]:  # Limit for speed
                pairs_to_try.append((mi, ci))
            for ci in cycle_bot_idx[:5]:
                pairs_to_try.append((mi, ci))
        
        # Add some random pairs
        for _ in range(50):
            i, j = rng.choice(n_edges, size=2, replace=False)
            if i != j:
                pairs_to_try.append((int(i), int(j)))
        
        # Deduplicate
        pairs_to_try = list(set(pairs_to_try))
        
        # Shuffle for variety
        rng.shuffle(pairs_to_try)
        
        for i, j in pairs_to_try:
            if time.time() > deadline - 2.0:
                break
            
            # Flip both edges
            new_signs = current_signs.copy()
            new_signs[i] *= -1
            new_signs[j] *= -1
            
            if not check_valid_fast(new_signs):
                continue
            
            new_radius = spectral_radius(new_signs)
            
            if new_radius < current_radius - improvement_threshold:
                current_signs = new_signs
                current_radius = new_radius
                improved = True
                no_improve_count = 0
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = current_signs.copy()
            else:
                no_improve_count += 1
        
        # Also try single flips
        if time.time() > deadline - 2.0:
            break
        
        single_improved = False
        for i in all_indices:
            if time.time() > deadline - 1.0:
                break
            
            new_signs = current_signs.copy()
            new_signs[i] *= -1
            
            if not check_valid_fast(new_signs):
                continue
            
            new_radius = spectral_radius(new_signs)
            
            if new_radius < current_radius - improvement_threshold:
                current_signs = new_signs
                current_radius = new_radius
                single_improved = True
                no_improve_count = 0
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = current_signs.copy()
        
        if not improved and not single_improved:
            # Try random restarts occasionally
            if no_improve_count % 500 == 0 and time.time() < deadline - 10.0:
                for _ in range(100):
                    signs = rng.choice([-1, 1], size=n_edges)
                    if check_valid_fast(signs):
                        r = spectral_radius(signs)
                        if r < best_radius:
                            best_radius = r
                            best_signs = signs.copy()
                            current_signs = signs.copy()
                            current_radius = r
                            no_improve_count = 0
                            break
    
    return best_signs
# EVOLVE-BLOCK-END
