"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s (or 90s) limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency structure for quick lookup
    # For prism graph: two m-cycles + matching
    # Vertices 0..m-1 form one cycle, m..2m-1 form the other
    # Matching: (i, i+m) for i in 0..m-1
    # Cycle edges: (i, (i+1)%m) and (m+i, m+(i+1)%m)
    # Matching edges: (i, i+m)
    
    # Map edge index to (u, v)
    edge_map = {}
    for idx, (u, v) in enumerate(edges):
        edge_map[(u, v)] = idx
        edge_map[(v, u)] = idx
    
    def get_edge_idx(u, v):
        return edge_map[(u, v)]
    
    # Identify edge types
    cycle_edges_top = []
    cycle_edges_bot = []
    match_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            cycle_edges_top.append(idx)
        elif u >= m and v >= m:
            cycle_edges_bot.append(idx)
        else:
            match_edges.append(idx)
    
    def build_signed_adjacency(signs):
        """Build the signed adjacency matrix of the base graph."""
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(A):
        """Compute spectral radius (max absolute eigenvalue)."""
        try:
            eigvals = np.linalg.eigvalsh(A)
            return np.max(np.abs(eigvals))
        except:
            return float('inf')
    
    def check_connectivity(signs):
        """
        Check if the 2-lift is connected using cycle product method.
        The lift has 2n vertices: (v, 0) and (v, 1).
        A sign of +1 means (u,0)-(v,0) and (u,1)-(v,1)
        A sign of -1 means (u,0)-(v,1) and (u,1)-(v,0)
        
        The lift is connected iff there's a path in the base graph
        whose product of signs is -1 (i.e., the fundamental group
        of the lift has order 2, not 1).
        
        More precisely: the lift is connected iff the subgroup generated
        by the cycle products (in {+1, -1}) is {+1, -1}, i.e., at least
        one cycle has product -1.
        """
        # BFS on the base graph, tracking the "side" parity
        # Start from vertex 0, side 0
        visited = set()
        visited.add((0, 0))
        queue = [(0, 0)]
        
        while queue:
            v, side = queue.pop(0)
            for u, w in edges:
                if u == v:
                    other = w
                elif w == v:
                    other = u
                else:
                    continue
                s = signs[get_edge_idx(v, other)]
                new_side = side if s == 1 else (1 - side)
                if (other, new_side) not in visited:
                    visited.add((other, new_side))
                    queue.append((other, new_side))
        
        # Connected iff we visited all 2n vertices
        return len(visited) == 2 * n
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        # For prism graphs, the base is 3-regular, so the lift is always 3-regular
        # if signs are valid. Just check connectivity.
        return check_connectivity(signs)
    
    def get_initial_signs_analytic():
        """
        Analytic initialization: maximize frustration.
        For ladder/prism cycles, set signs so that each 4-cycle (ladder)
        has product -1 (max frustration).
        
        Strategy: set all top cycle edges to +1, all bottom cycle edges to +1,
        and all matching edges to -1. This makes each 4-cycle (i, i+1, i+1+m, i+m)
        have product: +1 * +1 * -1 * -1 = +1... that's not frustrated.
        
        Let me think again. A 4-cycle in the prism: i -> i+1 (top), i+1 -> i+1+m (match),
        i+1+m -> i+m (bottom, reversed), i+m -> i (match, reversed).
        Product = s_top[i] * s_match[i+1] * s_bot[i+1] * s_match[i]
        
        For frustration, we want product = -1.
        
        Alternative: set top cycle to +1, bottom cycle to -1, matching to +1.
        Product = 1 * 1 * (-1) * 1 = -1. Yes!
        
        Actually let me be more careful with edge indexing.
        """
        signs = np.ones(num_edges, dtype=np.int8)
        
        # Set bottom cycle edges to -1
        for idx in cycle_edges_bot:
            signs[idx] = -1
        
        # Top cycle and matching stay +1
        return signs
    
    def get_random_signs(rng):
        """Generate random signs."""
        return rng.choice([-1, 1], size=num_edges).astype(np.int8)
    
    # Start with analytic initialization
    best_signs = get_initial_signs_analytic()
    if not is_valid_lift(best_signs):
        # Try to fix: flip one matching edge
        for idx in match_edges:
            trial = best_signs.copy()
            trial[idx] *= -1
            if is_valid_lift(trial):
                best_signs = trial
                break
        else:
            # Fallback to random
            rng = np.random.default_rng(3)
            for _ in range(1000):
                trial = get_random_signs(rng)
                if is_valid_lift(trial):
                    best_signs = trial
                    break
            else:
                # Shouldn't happen for prism graphs
                best_signs = get_random_signs(rng)
    
    best_radius = spectral_radius(build_signed_adjacency(best_signs))
    
    # Local search: flip one edge at a time
    def local_search(signs, radius, rng, time_limit):
        """Flip one edge at a time to reduce spectral radius."""
        signs = signs.copy()
        radius = radius
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try flipping each edge
            for i in range(num_edges):
                trial = signs.copy()
                trial[i] *= -1
                if not is_valid_lift(trial):
                    continue
                A = build_signed_adjacency(trial)
                r = spectral_radius(A)
                if r < radius - 1e-10:
                    signs = trial
                    radius = r
                    improved = True
                    break  # Restart from this point
            # Also try random flips
            for _ in range(10):
                i = rng.integers(0, num_edges)
                trial = signs.copy()
                trial[i] *= -1
                if not is_valid_lift(trial):
                    continue
                A = build_signed_adjacency(trial)
                r = spectral_radius(A)
                if r < radius - 1e-10:
                    signs = trial
                    radius = r
                    improved = True
                    break
        return signs, radius
    
    # Run local search with time management
    rng = np.random.default_rng(3)
    
    # First, do a thorough local search on the analytic init
    best_signs, best_radius = local_search(best_signs, best_radius, rng, time.time() + 30.0)
    
    # If time remains, try multiple random restarts
    while time.time() < deadline - 5.0:
        trial = get_random_signs(rng)
        if not is_valid_lift(trial):
            continue
        trial_radius = spectral_radius(build_signed_adjacency(trial))
        
        if trial_radius < best_radius:
            best_signs = trial
            best_radius = trial_radius
        
        # Local search from this random start
        remaining = deadline - 5.0 - time.time()
        if remaining > 2.0:
            trial_signs, trial_radius = local_search(trial, trial_radius, rng, time.time() + min(remaining, 10.0))
            if trial_radius < best_radius:
                best_signs = trial_signs
                best_radius = trial_radius
        
        # Check time
        if time.time() > deadline - 3.0:
            break
    
    # Final check: make sure we have a valid result
    if not is_valid_lift(best_signs):
        # Emergency: find any valid lift
        for _ in range(10000):
            trial = get_random_signs(rng)
            if is_valid_lift(trial):
                best_signs = trial
                break
    
    return best_signs.astype(np.int8)
# EVOLVE-BLOCK-END
