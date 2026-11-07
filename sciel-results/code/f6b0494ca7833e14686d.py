import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    def build_signed_adj(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        num_lift_vertices = 2 * n
        parent = list(range(num_lift_vertices))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                union(u * 2, v * 2)
                union(u * 2 + 1, v * 2 + 1)
            else:
                union(u * 2, v * 2 + 1)
                union(u * 2 + 1, v * 2)
        root = find(0)
        for i in range(1, num_lift_vertices):
            if find(i) != root:
                return False
        return True
    
    def local_search(signs, max_iterations=50):
        """Try flipping individual signs to reduce spectral radius."""
        current_signs = signs.copy()
        current_radius = spectral_radius(current_signs)
        if not is_connected_lift(current_signs):
            return signs, float('inf')
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            improved = False
            for idx in range(num_edges):
                # Try flipping this sign
                trial_signs = current_signs.copy()
                trial_signs[idx] *= -1
                if not is_connected_lift(trial_signs):
                    continue
                trial_radius = spectral_radius(trial_signs)
                if trial_radius < current_radius - 1e-10:
                    current_signs = trial_signs
                    current_radius = trial_radius
                    improved = True
                    break
            if not improved:
                break
        
        return current_signs, current_radius
    
    def make_signing(target_ladders, target_top):
        """Construct signing with matching edges = +1 given ladder and top cycle products."""
        signs = np.zeros(num_edges, dtype=int)
        # Top edges: first m-1 are +1, last is target_top
        for i in range(m - 1):
            signs[i] = 1
        signs[m - 1] = target_top
        # Bottom edges from ladder constraints
        for i in range(m):
            ip = (i + 1) % m
            signs[m + ip] = target_ladders[i] * signs[i]
        # Matching edges = +1
        for i in range(m):
            signs[2*m + i] = 1
        return signs
    
    best_signs = None
    best_radius = float('inf')
    
    # Strategy: try the "balanced" configuration first (all ladder products = -1),
    # then enumerate all switching classes.
    
    # For odd m, target_top = -1 gives the bottom cycle product = (-1)^m * (-1) = 1 * (-1) = -1
    # For even m, target_top = -1 gives bottom cycle product = (-1)^m * (-1) = 1 * (-1) = -1
    # Wait: bottom_prod = product(target_ladders) * product(top_signs) = (-1)^m * target_top
    # For m=4: (-1)^4 * (-1) = -1
    # For m=5: (-1)^5 * (-1) = 1
    # For m=7: (-1)^7 * (-1) = 1
    # For m=9: (-1)^9 * (-1) = 1
    
    # Try balanced: all ladders = -1, top = -1
    target_ladders_balanced = [-1] * m
    target_top_balanced = -1
    signs_balanced = make_signing(target_ladders_balanced, target_top_balanced)
    if is_connected_lift(signs_balanced):
        r = spectral_radius(signs_balanced)
        if r < best_radius:
            best_radius = r
            best_signs = signs_balanced.copy()
    
    # Now enumerate all 2^(m+1) switching classes
    for pattern_idx in range(2 ** (m + 1)):
        if time.time() > deadline - 5:
            break
        
        bits = [(pattern_idx >> k) & 1 for k in range(m + 1)]
        target_ladders = [1 if b == 0 else -1 for b in bits[:m]]
        target_top = 1 if bits[m] == 0 else -1
        
        signs = make_signing(target_ladders, target_top)
        
        if not is_connected_lift(signs):
            continue
        
        r = spectral_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    # Local search from best found so far
    if best_signs is not None and time.time() < deadline - 15:
        polished_signs, polished_radius = local_search(best_signs, max_iterations=30)
        if polished_radius < best_radius:
            best_signs = polished_signs
            best_radius = polished_radius
    
    # Try local search from a few other promising starting points
    if best_signs is not None and time.time() < deadline - 20:
        # Try from all-+1 configuration
        all_plus = np.ones(num_edges, dtype=int)
        if is_connected_lift(all_plus):
            polished_signs, polished_radius = local_search(all_plus, max_iterations=20)
            if polished_radius < best_radius:
                best_signs = polished_signs
                best_radius = polished_radius
        
        # Try from all -1 configuration
        all_minus = -np.ones(num_edges, dtype=int)
        if is_connected_lift(all_minus):
            polished_signs, polished_radius = local_search(all_minus, max_iterations=20)
            if polished_radius < best_radius:
                best_signs = polished_signs
                best_radius = polished_radius
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
