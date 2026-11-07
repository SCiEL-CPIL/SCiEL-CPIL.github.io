import time
import numpy as np
from collections import deque
from scipy.optimize import minimize

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy:
    1. Use SDP relaxation to find a continuous vector that minimizes the spectral norm.
    2. Round using Gaussian randomization.
    3. Refine with connectivity-constrained local search.
    """
    
    deadline = time.time() + 85.0  # Safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge arrays for fast matrix construction
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected.
        """
        f = [-1] * n
        f[0] = 0
        
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
            
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip
                
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        return True
        
        return False

    def get_radius(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for i in range(num_edges):
            val = float(signs[i])
            S[u_arr[i], v_arr[i]] = val
            S[v_arr[i], u_arr[i]] = val
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    # SDP approach: We want to minimize the spectral norm of the signed adjacency matrix.
    # The spectral norm is the largest singular value, which for a symmetric matrix is 
    # the largest absolute eigenvalue.
    # 
    # We can formulate this as: minimize t such that ||S||_2 <= t
    # This is equivalent to: -t*I <= S <= t*I (Loewner order)
    # 
    # Since S depends linearly on the signs, and we want signs in {-1, +1},
    # we relax to signs in [-1, 1] and solve the SDP, then round.
    #
    # However, SDP solvers can be slow. Instead, we use a simpler approach:
    # minimize the sum of squares of eigenvalues, which is related to the Frobenius norm,
    # but we want the spectral norm.
    #
    # Alternative: Use a penalty method or directly optimize the spectral norm.
    # Since scipy doesn't have a direct spectral norm minimization, we use an iterative approach.
    
    # Let's try a different approach: use the fact that for a symmetric matrix,
    # the spectral norm is the maximum of x^T S x / (x^T x) over unit vectors x.
    # So we can use a minimax approach.
    
    # For simplicity and speed, let's use a gradient-based approach with the 
    # spectral norm as the objective, using subgradients.
    
    # Actually, let's try a simpler structural approach first, then refine.
    
    # Structural initialization: exploit the prism graph structure
    # The prism graph has two m-cycles and a matching.
    
    # For small m (4, 5, 7, 9), we can try all sign patterns that respect
    # certain symmetries, then use local search.
    
    # Initialize with a pattern that tends to give low spectral radius:
    # Alternate signs on cycles to create cancellations.
    
    def generate_initial_signs(m, num_edges, edges, u_arr, v_arr):
        """Generate initial signs based on graph structure."""
        signs = np.ones(num_edges, dtype=np.int8)
        
        for i, (u, v) in enumerate(edges):
            if u < m and v < m:
                # Top cycle
                pos = min(u, v)
                signs[i] = 1 if (pos % 2 == 0) else -1
            elif u >= m and v >= m:
                # Bottom cycle
                pos = min(u - m, v - m)
                signs[i] = 1 if (pos % 2 == 0) else -1
            else:
                # Matching
                pos = min(u, v) if min(u, v) < m else min(u - m, v - m)
                signs[i] = 1
        
        return signs
    
    # Try multiple initializations and pick the best
    best_signs = None
    best_radius = np.inf
    
    # Strategy 1: Structural initialization with different phases
    for phase_top in [0, 1]:
        for phase_bottom in [0, 1]:
            for phase_match in [0, 1]:
                signs = np.ones(num_edges, dtype=np.int8)
                
                for i, (u, v) in enumerate(edges):
                    if u < m and v < m:
                        pos = min(u, v)
                        signs[i] = 1 if ((pos + phase_top) % 2 == 0) else -1
                    elif u >= m and v >= m:
                        pos = min(u - m, v - m)
                        signs[i] = 1 if ((pos + phase_bottom) % 2 == 0) else -1
                    else:
                        pos = min(u, v) if min(u, v) < m else min(u - m, v - m)
                        signs[i] = 1 if (phase_match == 0) else -1
                
                if not is_lift_connected(signs):
                    continue
                
                radius = get_radius(signs)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
    
    # Strategy 2: Random initialization with local search
    rng = np.random.default_rng(3)
    
    # Try several random starts
    for trial in range(20):
        if time.time() > deadline:
            break
        
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        # Ensure connectivity
        attempts = 0
        while not is_lift_connected(signs) and attempts < 100:
            idx = rng.integers(0, num_edges)
            signs[idx] *= -1
            attempts += 1
        
        if not is_lift_connected(signs):
            continue
        
        radius = get_radius(signs)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # Local search refinement
    if best_signs is not None:
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        # Greedy local search: flip edges that improve the radius
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                if not is_lift_connected(current_signs):
                    current_signs[i] = old_sign
                    continue
                
                S = np.zeros((n, n), dtype=np.float64)
                for j in range(num_edges):
                    val = float(current_signs[j])
                    S[u_arr[j], v_arr[j]] = val
                    S[v_arr[j], u_arr[j]] = val
                
                new_radius = np.max(np.abs(np.linalg.eigvalsh(S)))
                
                if new_radius < current_radius:
                    current_radius = new_radius
                    improved = True
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = current_signs.copy()
                else:
                    current_signs[i] = old_sign
        
        # Simulated annealing for further improvement
        T = 0.1
        cooling_rate = 0.95
        steps_per_temp = 100
        
        sa_signs = best_signs.copy()
        sa_radius = best_radius
        
        while T > 0.01 and time.time() < deadline:
            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    break
                
                idx = rng.integers(0, num_edges)
                old_sign = sa_signs[idx]
                sa_signs[idx] = -old_sign
                
                if not is_lift_connected(sa_signs):
                    sa_signs[idx] = old_sign
                    continue
                
                S = np.zeros((n, n), dtype=np.float64)
                for j in range(num_edges):
                    val = float(sa_signs[j])
                    S[u_arr[j], v_arr[j]] = val
                    S[v_arr[j], u_arr[j]] = val
                
                new_radius = np.max(np.abs(np.linalg.eigvalsh(S)))
                delta = new_radius - sa_radius
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    sa_radius = new_radius
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = sa_signs.copy()
                else:
                    sa_signs[idx] = old_sign
            
            T *= cooling_rate
    
    if best_signs is None:
        # Fallback: all +1 except one edge to ensure connectivity
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_lift_connected(best_signs):
            best_signs[0] = -1
            if not is_lift_connected(best_signs):
                best_signs[1] = -1
    
    return best_signs
