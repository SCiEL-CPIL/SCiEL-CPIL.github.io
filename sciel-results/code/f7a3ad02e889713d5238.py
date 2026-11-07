import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy:
    1. Precompute adjacency list for fast connectivity checks.
    2. Use continuous phase-space optimization exploiting prism circulant structure.
    3. Multi-restart local search with rank-1 spectral updates for speed.
    4. Final high-precision polish with triple-flip moves.
    """
    
    deadline = time.time() + 82.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute adjacency list once
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected (signing is NOT a coboundary).
        """
        f = [-1] * n
        f[0] = 0
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

    def build_matrix(signs):
        S = np.zeros((n, n), dtype=np.float64)
        S[u_arr, v_arr] = signs.astype(np.float64)
        S[v_arr, u_arr] = signs.astype(np.float64)
        return S

    def get_radius(signs):
        S = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    def phase_optimization(num_iter=200, num_restarts=10):
        """
        Optimize continuous phases for each edge to minimize spectral radius.
        For prism graphs, we can parameterize signs as cos(theta_i) where theta_i in [0, pi].
        This exploits the circulant structure to find balanced sign assignments.
        """
        best_x = None
        best_radius = np.inf
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            
            rng_phase = np.random.default_rng(restart * 7 + 13)
            # Initialize phases randomly in [0, pi]
            theta = rng_phase.uniform(0, np.pi, size=num_edges)
            v = np.zeros_like(theta)  # momentum
            lr = 0.1
            best_x_local = None
            best_radius_local = np.inf
            
            for it in range(num_iter):
                if time.time() > deadline - 8.0:
                    break
                
                # Convert phases to soft signs: x_i = cos(theta_i)
                x = np.cos(theta)
                
                # Build matrix with soft signs
                S = np.zeros((n, n), dtype=np.float64)
                S[u_arr, v_arr] = x
                S[v_arr, u_arr] = x
                
                # Compute spectral radius and gradient
                eigvals, eigvecs = np.linalg.eigh(S)
                max_idx = np.argmax(np.abs(eigvals))
                v_max = eigvecs[:, max_idx]
                
                # Gradient of ||S||_2 w.r.t. S is (v_max v_max^T) / ||S||_2
                grad_S = np.outer(v_max, v_max)
                
                # Gradient w.r.t. x_i: grad_S[u_i, v_i] + grad_S[v_i, u_i]
                grad_x = grad_S[u_arr, v_arr] + grad_S[v_arr, u_arr]
                
                # Chain rule: d(x_i)/d(theta_i) = -sin(theta_i)
                # So grad_theta_i = grad_x_i * (-sin(theta_i))
                sin_theta = np.sin(theta)
                grad_theta = -grad_x * sin_theta
                
                # Momentum update
                v = 0.9 * v - lr * grad_theta
                theta = theta + v
                # Wrap theta to [0, pi]
                theta = np.mod(theta, np.pi)
                
                # Periodically check connectivity with rounded signs
                if it % 5 == 0:
                    rounded = np.sign(x)
                    rounded[rounded == 0] = 1
                    if is_lift_connected(rounded):
                        r = get_radius(rounded)
                        if r < best_radius_local:
                            best_radius_local = r
                            best_x_local = x.copy()
            
            if best_x_local is not None and best_radius_local < best_radius:
                best_radius = best_radius_local
                best_x = best_x_local.copy()
        
        if best_x is None:
            best_x = np.ones(num_edges)
        
        return best_x, best_radius

    def sdp_rounding(x, num_rounds=100):
        """
        Round soft signs to discrete signs using Gaussian randomization.
        """
        rng = np.random.default_rng(7)
        best_signs = None
        best_radius = np.inf
        
        for _ in range(num_rounds):
            # Gaussian randomization: sign = sign(x_i + noise)
            noise = rng.normal(0, 1, size=num_edges)
            rounded = np.sign(x + noise)
            rounded[rounded == 0] = 1
            
            if is_lift_connected(rounded):
                r = get_radius(rounded)
                if r < best_radius:
                    best_radius = r
                    best_signs = rounded.copy()
        
        return best_signs, best_radius

    def local_search(signs, max_time=5.0):
        """
        Local search with single-flip and double-flip moves.
        Returns (best_signs, best_radius) found.
        """
        if not is_lift_connected(signs):
            return None, np.inf
        
        current_S = build_matrix(signs)
        current_radius = np.max(np.abs(np.linalg.eigvalsh(current_S)))
        local_best_signs = signs.copy()
        local_best_radius = current_radius
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 80
        local_rng = np.random.default_rng(hash(tuple(signs)) % (2**32))
        
        while time.time() - start_time < max_time and no_improve_count < max_no_improve:
            if time.time() > deadline - 2.0:
                break
            improved = False
            
            # Single-flip pass
            for i in range(num_edges):
                if time.time() > deadline - 1.5:
                    return local_best_signs, local_best_radius
                
                old_sign = signs[i]
                new_sign = -old_sign
                signs[i] = new_sign
                
                if not is_lift_connected(signs):
                    signs[i] = old_sign
                    continue
                
                # Fast radius computation
                S_new = current_S.copy()
                u, v = edges[i]
                delta = new_sign - old_sign
                S_new[u, v] += delta
                S_new[v, u] += delta
                new_radius = np.max(np.abs(np.linalg.eigvalsh(S_new)))
                
                if new_radius < current_radius - 1e-12:
                    current_S = S_new
                    current_radius = new_radius
                    improved = True
                    no_improve_count = 0
                    if new_radius < local_best_radius:
                        local_best_radius = new_radius
                        local_best_signs = signs.copy()
                else:
                    signs[i] = old_sign
            
            # Double-flip pass (every few iterations to escape local minima)
            if not improved and no_improve_count >= 3:
                for _ in range(num_edges * 2):
                    if time.time() > deadline - 1.5:
                        return local_best_signs, local_best_radius
                    
                    i = local_rng.integers(0, num_edges)
                    j = local_rng.integers(0, num_edges)
                    if i == j:
                        continue
                    
                    old_si, old_sj = signs[i], signs[j]
                    signs[i] = -old_si
                    signs[j] = -old_sj
                    
                    if not is_lift_connected(signs):
                        signs[i] = old_si
                        signs[j] = old_sj
                        continue
                    
                    S_new = current_S.copy()
                    ui, vi = edges[i]
                    uj, vj = edges[j]
                    S_new[ui, vi] -= 2 * old_si
                    S_new[vi, ui] -= 2 * old_si
                    S_new[uj, vj] -= 2 * old_sj
                    S_new[vj, uj] -= 2 * old_sj
                    new_radius = np.max(np.abs(np.linalg.eigvalsh(S_new)))
                    
                    if new_radius < current_radius - 1e-12:
                        current_S = S_new
                        current_radius = new_radius
                        improved = True
                        no_improve_count = 0
                        if new_radius < local_best_radius:
                            local_best_radius = new_radius
                            local_best_signs = signs.copy()
                    else:
                        signs[i] = old_si
                        signs[j] = old_sj
            
            if not improved:
                no_improve_count += 1
        
        return local_best_signs, local_best_radius

    best_signs = None
    best_radius = np.inf
    
    rng = np.random.default_rng(3)
    
    # Strategy 1: Exhaustive Search for small m
    if num_edges <= 15:
        # m=4: 12 edges, 2^12 = 4096
        # m=5: 15 edges, 2^15 = 32768
        max_combo = 1 << num_edges
        for k in range(max_combo):
            if time.time() > deadline - 2.0:
                break
            signs = np.where((k >> np.arange(num_edges)) & 1, 1, -1).astype(np.int8)
            if not is_lift_connected(signs):
                continue
            S = build_matrix(signs)
            eigvals = np.linalg.eigvalsh(S)
            radius = np.max(np.abs(eigvals))
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        if best_signs is not None:
            return best_signs

    # Strategy 2: Phase-space optimization with multiple rounds
    if time.time() < deadline - 15.0:
        phase_x, phase_radius = phase_optimization(num_iter=150, num_restarts=8)
        if phase_radius < best_radius:
            best_radius = phase_radius
            # Round the phase solution
            rounded_signs, rounded_radius = sdp_rounding(phase_x, num_rounds=80)
            if rounded_signs is not None and rounded_radius < best_radius:
                best_radius = rounded_radius
                best_signs = rounded_signs.copy()
            
            # Use rounded signs as starting point for local search
            if rounded_signs is not None:
                ls_signs, ls_radius = local_search(rounded_signs.copy(), max_time=4.0)
                if ls_signs is not None and ls_radius < best_radius:
                    best_radius = ls_radius
                    best_signs = ls_signs.copy()

    # Strategy 3: Multi-restart local search with single and double flips
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 6.0:
            break
        
        # Generate initial random signs
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        # Ensure connectivity
        attempts = 0
        while not is_lift_connected(signs) and attempts < 50:
            idx = rng.integers(0, num_edges)
            signs[idx] *= -1
            attempts += 1
        if not is_lift_connected(signs):
            continue
        
        # Run local search with adaptive time
        remaining = deadline - 6.0 - time.time()
        ls_time = min(4.0, max(1.0, remaining / (num_restarts - r)))
        ls_signs, ls_radius = local_search(signs, max_time=ls_time)
        
        if ls_signs is not None and ls_radius < best_radius:
            best_radius = ls_radius
            best_signs = ls_signs.copy()
    
    # Final high-precision polish: exhaustive neighborhood search around best
    if best_signs is not None and time.time() < deadline - 4.0:
        # Multiple passes of single and double flips with tighter convergence
        for pass_num in range(5):
            if time.time() > deadline - 2.0:
                break
            
            # Try all single flips
            for i in range(num_edges):
                if time.time() > deadline - 1.5:
                    break
                old = best_signs[i]
                best_signs[i] = -old
                if not is_lift_connected(best_signs):
                    best_signs[i] = old
                    continue
                S_new = build_matrix(best_signs)
                r = np.max(np.abs(np.linalg.eigvalsh(S_new)))
                if r < best_radius - 1e-12:
                    best_radius = r
                else:
                    best_signs[i] = old
            
            # Try all double flips
            for i in range(num_edges):
                if time.time() > deadline - 1.5:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() > deadline - 1.0:
                        break
                    si, sj = best_signs[i], best_signs[j]
                    best_signs[i] = -si
                    best_signs[j] = -sj
                    if not is_lift_connected(best_signs):
                        best_signs[i] = si
                        best_signs[j] = sj
                        continue
                    S_new = build_matrix(best_signs)
                    r = np.max(np.abs(np.linalg.eigvalsh(S_new)))
                    if r < best_radius - 1e-12:
                        best_radius = r
                        # Keep the flip
                        si, sj = best_signs[i], best_signs[j]
                    else:
                        best_signs[i] = si
                        best_signs[j] = sj
    
    if best_signs is None:
        # Fallback
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_lift_connected(best_signs):
            best_signs[0] = -1
            if not is_lift_connected(best_signs):
                best_signs[1] = -1

    return best_signs
