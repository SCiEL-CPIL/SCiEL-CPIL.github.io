import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n = graph['n']
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
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
        return np.max(np.abs(eigvals))
    
    def is_valid(signs):
        try:
            check_lift(graph, signs)
            return True
        except Exception:
            return False
    
    best_signs = None
    best_radius = float('inf')
    
    start_time = time.time()
    time_limit = 14
    
    rng = np.random.default_rng(42)
    
    # Identify edge types
    edge_type = np.zeros(n_edges, dtype=int)
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            edge_type[i] = 0
        elif u >= m and v >= m:
            edge_type[i] = 1
        else:
            edge_type[i] = 2
    
    # Brute force for small graphs
    if n_edges <= 16:
        for combo in range(2 ** n_edges):
            if time.time() - start_time > time_limit:
                break
            signs = np.array([(combo >> i) & 1 for i in range(n_edges)], dtype=int)
            signs = 2 * signs - 1
            if not is_valid(signs):
                continue
            radius = spectral_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    else:
        # Local search for larger graphs
        candidates = [
            np.ones(n_edges, dtype=int),
            -np.ones(n_edges, dtype=int),
        ]
        c1 = np.ones(n_edges, dtype=int)
        c1[edge_type == 2] = -1
        candidates.append(c1)
        c2 = np.ones(n_edges, dtype=int)
        c2[edge_type != 2] = -1
        candidates.append(c2)
        for _ in range(20):
            candidates.append(rng.choice([-1, 1], size=n_edges))
        
        for signs in candidates:
            if time.time() - start_time > time_limit:
                break
            if not is_valid(signs):
                continue
            radius = spectral_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
            
            current_signs = signs.copy()
            current_radius = radius
            T = 0.2
            
            for iteration in range(1000):
                if time.time() - start_time > time_limit:
                    break
                idx = rng.integers(0, n_edges)
                current_signs[idx] = -current_signs[idx]
                if not is_valid(current_signs):
                    current_signs[idx] = -current_signs[idx]
                    continue
                new_radius = spectral_radius(current_signs)
                delta = new_radius - current_radius
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
                else:
                    current_signs[idx] = -current_signs[idx]
                T *= 0.99
    
    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=int)
    
    return best_signs
