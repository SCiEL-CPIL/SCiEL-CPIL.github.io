import numpy as np
import time
import random

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Precompute edge indices for fast lookup
    # Edge types: 0..m-1 are bottom cycle, m..2m-1 are ladder, 2m..3m-1 are top cycle
    # But we need to verify the actual edge ordering from the graph
    
    # Build a lookup for edge positions by type
    bottom_cycle_edges = []
    ladder_edges = []
    top_cycle_edges = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_cycle_edges.append(idx)
        elif u >= m and v >= m:
            top_cycle_edges.append(idx)
        else:
            ladder_edges.append(idx)
    
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
    
    def local_search(signs, max_iterations=100, tol=1e-10):
        """Try flipping individual signs to reduce spectral radius."""
        current_signs = signs.copy()
        if not is_connected_lift(current_signs):
            return signs, float('inf')
        current_radius = spectral_radius(current_signs)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            improved = False
            # Shuffle edge order for diversity
            edge_order = list(range(num_edges))
            random.shuffle(edge_order)
            for idx in edge_order:
                trial_signs = current_signs.copy()
                trial_signs[idx] *= -1
                if not is_connected_lift(trial_signs):
                    continue
                trial_radius = spectral_radius(trial_signs)
                if trial_radius < current_radius - tol:
                    current_signs = trial_signs
                    current_radius = trial_radius
                    improved = True
                    break
            if not improved:
                break
        
        return current_signs, current_radius
    
    def simulated_annealing(signs, max_iters=200, T_start=1.0, T_end=0.01):
        """Simulated annealing to escape local optima."""
        current_signs = signs.copy()
        if not is_connected_lift(current_signs):
            return signs, float('inf')
        current_radius = spectral_radius(current_signs)
        best_signs = current_signs.copy()
        best_radius = current_radius
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            T *= cooling
            
            # Pick random edge to flip
            idx = random.randint(0, num_edges - 1)
            trial_signs = current_signs.copy()
            trial_signs[idx] *= -1
            
            if not is_connected_lift(trial_signs):
                continue
            
            trial_radius = spectral_radius(trial_signs)
            delta = trial_radius - current_radius
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current_signs = trial_signs
                current_radius = trial_radius
                if current_radius < best_radius - 1e-10:
                    best_signs = current_signs.copy()
                    best_radius = current_radius
        
        return best_signs, best_radius
    
    def make_signing(target_ladders, target_top):
        """Construct signing with matching edges = +1 given ladder and top cycle products."""
        signs = np.ones(num_edges, dtype=int)
        
        # Set top cycle signs: first m-1 edges are +1, last is target_top
        # Need to map top_cycle_edges to the correct positions
        # top_cycle_edges should be in order around the cycle
        for i in range(m - 1):
            signs[top_cycle_edges[i]] = 1
        signs[top_cycle_edges[m - 1]] = target_top
        
        # Set ladder signs based on target_ladders and bottom cycle signs (all +1)
        for i in range(m):
            # ladder_edges[i] connects vertex i (bottom) to vertex i+m (top)
            # The sign should be target_ladders[i]
            signs[ladder_edges[i]] = target_ladders[i]
        
        # Bottom cycle edges are all +1 (already set)
        return signs
    
    def make_signing_general(bottom_cycle_signs, ladder_signs, top_cycle_signs):
        """Construct signing with arbitrary cycle and ladder signs."""
        signs = np.ones(num_edges, dtype=int)
        for i in range(m):
            signs[bottom_cycle_edges[i]] = bottom_cycle_signs[i]
            signs[ladder_edges[i]] = ladder_signs[i]
            signs[top_cycle_edges[i]] = top_cycle_signs[i]
        return signs
    
    best_signs = None
    best_radius = float('inf')
    
    # Enumerate all 2^(m+1) switching classes
    # The spectrum is switching-invariant, so we can fix bottom cycle to all +1
    # and enumerate ladder signs and top cycle product
    for pattern_idx in range(2 ** (m + 1)):
        if time.time() > deadline - 20:
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
    if best_signs is not None and time.time() < deadline - 20:
        polished_signs, polished_radius = local_search(best_signs, max_iterations=50)
        if polished_radius < best_radius:
            best_signs = polished_signs
            best_radius = polished_radius
    
    # Simulated annealing from best
    if best_signs is not None and time.time() < deadline - 25:
        sa_signs, sa_radius = simulated_annealing(best_signs, max_iters=150)
        if sa_radius < best_radius - 1e-10:
            best_signs = sa_signs
            best_radius = sa_radius
        # Follow up with local search
        if time.time() < deadline - 15:
            polished_signs, polished_radius = local_search(best_signs, max_iterations=50)
            if polished_radius < best_radius:
                best_signs = polished_signs
                best_radius = polished_radius
    
    # Try more diverse starting points with local search
    if time.time() < deadline - 30:
        # All +1
        all_plus = np.ones(num_edges, dtype=int)
        if is_connected_lift(all_plus):
            polished_signs, polished_radius = local_search(all_plus, max_iterations=30)
            if polished_radius < best_radius:
                best_signs = polished_signs
                best_radius = polished_radius
        
        # All -1
        all_minus = -np.ones(num_edges, dtype=int)
        if is_connected_lift(all_minus):
            polished_signs, polished_radius = local_search(all_minus, max_iterations=30)
            if polished_radius < best_radius:
                best_signs = polished_signs
                best_radius = polished_radius
        
        # Random restarts
        rng = np.random.default_rng(42)
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            random_signs = rng.choice([-1, 1], size=num_edges)
            if not is_connected_lift(random_signs):
                continue
            polished_signs, polished_radius = local_search(random_signs, max_iterations=30)
            if polished_radius < best_radius:
                best_signs = polished_signs
                best_radius = polished_radius
            # Try simulated annealing from this point too
            if time.time() < deadline - 20:
                sa_signs, sa_radius = simulated_annealing(random_signs, max_iters=100)
                if sa_radius < best_radius - 1e-10:
                    best_signs = sa_signs
                    best_radius = sa_radius
    
    # Final polish with tighter tolerance
    if best_signs is not None and time.time() < deadline - 10:
        polished_signs, polished_radius = local_search(best_signs, max_iterations=50, tol=1e-12)
        if polished_radius < best_radius:
            best_signs = polished_signs
            best_radius = polished_radius
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
