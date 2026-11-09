import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Identify edge types
    c1_edges = []
    c2_edges = []
    m_edges = []

    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            c1_edges.append(i)
        elif u >= m and v >= m:
            c2_edges.append(i)
        else:
            m_edges.append(i)

    # Precompute edge endpoints
    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

    # Precompute lift adjacency contributions
    lift_adj_plus = [None] * num_edges
    lift_adj_minus = [None] * num_edges
    
    for i in range(num_edges):
        u = edge_u[i]
        v = edge_v[i]
        u0, u1 = u * 2, u * 2 + 1
        v0, v1 = v * 2, v * 2 + 1
        
        lift_adj_plus[i] = [(u0, v0), (v0, u0), (u1, v1), (v1, u1)]
        lift_adj_minus[i] = [(u0, v1), (v1, u0), (u1, v0), (v0, u1)]

    def evaluate_signs(signs):
        lift_n = 2 * n
        
        adj_lift = [[] for _ in range(lift_n)]
        for i in range(num_edges):
            if signs[i] == 1:
                for u_l, v_l in lift_adj_plus[i]:
                    adj_lift[u_l].append(v_l)
            else:
                for u_l, v_l in lift_adj_minus[i]:
                    adj_lift[u_l].append(v_l)
        
        visited = [False] * lift_n
        queue = deque([0])
        visited[0] = True
        while queue:
            u = queue.popleft()
            for v in adj_lift[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        if not all(visited):
            return float('inf')
            
        A = np.zeros((n, n), dtype=float)
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    best_radius = float('inf')
    best_signs = None

    def try_signs(signs):
        nonlocal best_radius, best_signs
        r = evaluate_signs(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    def get_cycle_edges_indices(edge_indices, offset=0):
        pos_map = {}
        for idx in edge_indices:
            u, v = edges[idx]
            u = u - offset
            v = v - offset
            if (u + 1) % m == v:
                pos = u
            elif (v + 1) % m == u:
                pos = v
            else:
                continue
            pos_map[pos] = idx
        return sorted(pos_map.items())

    c1_ordered = get_cycle_edges_indices(c1_edges, 0)
    c2_ordered = get_cycle_edges_indices(c2_edges, m)
    m_ordered = sorted(m_edges, key=lambda i: edges[i][0] if edges[i][0] < m else edges[i][1])

    # Generate initial candidates
    candidates = []
    
    # Pattern 1: Alt C1, Alt C2, -1 M
    s1 = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s1[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s1[idx] = 1 if pos % 2 == 0 else -1
    for idx in m_edges: s1[idx] = -1
    candidates.append(s1)
    
    # Pattern 2: Alt C1, Anti-Alt C2, -1 M
    s2 = s1.copy()
    for pos, idx in c2_ordered: s2[idx] = -1 if pos % 2 == 0 else 1
    candidates.append(s2)
    
    # Pattern 3: Alt C1, Alt C2, +1 M
    s3 = s1.copy()
    for idx in m_edges: s3[idx] = 1
    candidates.append(s3)
    
    # Pattern 4: All +1 C1, All +1 C2, Alt M
    s4 = np.ones(num_edges, dtype=int)
    for pos, idx in enumerate(m_ordered): s4[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s4)
    
    # Pattern 5: Alt C1, Alt C2, Alt M
    s5 = s1.copy()
    for pos, idx in enumerate(m_ordered): s5[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s5)
    
    for s in candidates:
        try_signs(s)

    # Phase 1: Single Flip Hill Climb
    current_signs = best_signs.copy()
    current_radius = best_radius

    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            current_signs[i] *= -1
            r = evaluate_signs(current_signs)
            if r < current_radius - 1e-9:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] *= -1
        if not improved:
            break

    # Phase 2: Pairwise Flip Search
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    max_pair_passes = 5
    for _ in range(max_pair_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            for j in range(i + 1, num_edges):
                if time.time() > deadline:
                    break
                current_signs[i] *= -1
                current_signs[j] *= -1
                r = evaluate_signs(current_signs)
                if r < current_radius - 1e-9:
                    current_radius = r
                    best_radius = r
                    best_signs = current_signs.copy()
                    improved = True
                    current_signs = best_signs.copy()
                    break
                else:
                    current_signs[i] *= -1
                    current_signs[j] *= -1
            if improved:
                break
        if not improved:
            break

    # Phase 3: Final Single Flip Polish
    current_signs = best_signs.copy()
    current_radius = best_radius
    for _ in range(10):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            current_signs[i] *= -1
            r = evaluate_signs(current_signs)
            if r < current_radius - 1e-9:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] *= -1
        if not improved:
            break

    # Phase 4: Phase Synchronization Optimization (Continuous Relaxation)
    # This is a structural change: instead of discrete flips, we optimize continuous phases
    # theta_v such that sign(Re(z_u conj(z_v))) gives the edge signs.
    # We minimize the spectral radius of the smooth matrix A_smooth where A_uv = tanh(k * cos(theta_u - theta_v)).
    # Then we map back to discrete signs and polish.
    
    def phase_sync_optimize(initial_signs, time_limit=15.0):
        # Initialize phases from initial signs if possible, otherwise random
        # For now, just use random phases with a fixed seed for reproducibility, 
        # but try a few different seeds.
        
        best_phase_signs = None
        best_phase_radius = float('inf')
        
        # Try multiple random initializations
        for seed in [42, 123, 456, 789, 101112]:
            if time.time() > start_time + time_limit:
                break
                
            rng = np.random.default_rng(seed)
            theta = rng.uniform(0, 2 * np.pi, size=n)
            
            # Gradient descent on theta to minimize spectral radius of smooth matrix
            # A_smooth[u,v] = tanh(k * cos(theta_u - theta_v)) for edges (u,v)
            # Objective: max |eigenvalue| of A_smooth
            # Gradient: d/dtheta_i max|eig| = ... (use finite differences or analytic)
            
            # For simplicity and speed, use finite differences with a small step
            # Or better: use the fact that for symmetric A, d lambda_max / d theta_i 
            # can be computed via eigenvectors.
            
            # Let's use a simple approach: 
            # 1. Compute A_smooth and its eigendecomposition
            # 2. The gradient of lambda_max w.r.t. A_uv is (v_max)_u * (v_max)_v
            # 3. d A_uv / d theta_i = -k * sin(theta_u - theta_v) * (delta_iu - delta_iv)
            
            k = 5.0  # Sharpness of tanh
            lr = 0.05  # Learning rate
            
            for step in range(100):
                if time.time() > start_time + time_limit:
                    break
                    
                # Build smooth adjacency matrix
                A_smooth = np.zeros((n, n), dtype=float)
                for i in range(num_edges):
                    u = edge_u[i]
                    v = edge_v[i]
                    val = np.tanh(k * np.cos(theta[u] - theta[v]))
                    A_smooth[u, v] = val
                    A_smooth[v, u] = val
                
                # Compute eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(A_smooth)
                
                # Find the eigenvalue with max absolute value
                max_idx = np.argmax(np.abs(eigvals))
                max_eig = eigvals[max_idx]
                v_max = eigvecs[:, max_idx]
                
                # Gradient of max |eig| w.r.t. A
                # If max_eig > 0, grad = v_max * v_max^T
                # If max_eig < 0, grad = -v_max * v_max^T (since d|lambda|/dA = sign(lambda) * v v^T)
                sign_eig = 1.0 if max_eig >= 0 else -1.0
                grad_A = sign_eig * np.outer(v_max, v_max)
                
                # Gradient w.r.t. theta
                grad_theta = np.zeros(n, dtype=float)
                for i in range(num_edges):
                    u = edge_u[i]
                    v = edge_v[i]
                    # d A_uv / d theta_u = -k * sin(theta_u - theta_v)
                    # d A_uv / d theta_v = k * sin(theta_u - theta_v)
                    sin_val = np.sin(theta[u] - theta[v])
                    grad_theta[u] += -k * sin_val * grad_A[u, v]
                    grad_theta[v] += k * sin_val * grad_A[u, v]
                
                # Update theta
                theta -= lr * grad_theta
                theta = theta % (2 * np.pi)  # Keep in [0, 2*pi)
                
                # Periodically evaluate the discrete signs from current theta
                if step % 20 == 0:
                    s_phase = np.ones(num_edges, dtype=int)
                    for i in range(num_edges):
                        u = edge_u[i]
                        v = edge_v[i]
                        if np.cos(theta[u] - theta[v]) < 0:
                            s_phase[i] = -1
                        else:
                            s_phase[i] = 1
                    
                    r = evaluate_signs(s_phase)
                    if r < best_phase_radius:
                        best_phase_radius = r
                        best_phase_signs = s_phase.copy()
            
            # Final evaluation
            s_phase = np.ones(num_edges, dtype=int)
            for i in range(num_edges):
                u = edge_u[i]
                v = edge_v[i]
                if np.cos(theta[u] - theta[v]) < 0:
                    s_phase[i] = -1
                else:
                    s_phase[i] = 1
            
            r = evaluate_signs(s_phase)
            if r < best_phase_radius:
                best_phase_radius = r
                best_phase_signs = s_phase.copy()
        
        return best_phase_signs

    # Run phase synchronization optimization
    phase_signs = phase_sync_optimize(best_signs, time_limit=15.0)
    if phase_signs is not None:
        try_signs(phase_signs)
        
        # Polish the phase-sync result with discrete local search
        if best_signs is not None and best_radius < float('inf'):
            current_signs = best_signs.copy()
            current_radius = best_radius
            
            # Quick single flip polish
            for _ in range(10):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(num_edges):
                    if time.time() > deadline:
                        break
                    current_signs[i] *= -1
                    r = evaluate_signs(current_signs)
                    if r < current_radius - 1e-9:
                        current_radius = r
                        best_radius = r
                        best_signs = current_signs.copy()
                        improved = True
                    else:
                        current_signs[i] *= -1
                if not improved:
                    break
            
            # Quick pairwise polish
            current_signs = best_signs.copy()
            current_radius = best_radius
            for _ in range(3):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(num_edges):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, num_edges):
                        if time.time() > deadline:
                            break
                        current_signs[i] *= -1
                        current_signs[j] *= -1
                        r = evaluate_signs(current_signs)
                        if r < current_radius - 1e-9:
                            current_radius = r
                            best_radius = r
                            best_signs = current_signs.copy()
                            improved = True
                            current_signs = best_signs.copy()
                            break
                        else:
                            current_signs[i] *= -1
                            current_signs[j] *= -1
                    if improved:
                        break
                if not improved:
                    break

    # Final fallback: if still inf, use a simple pattern
    if best_signs is None or best_radius == float('inf'):
        # Use pattern 1 as fallback
        s_fallback = np.ones(num_edges, dtype=int)
        for pos, idx in c1_ordered: s_fallback[idx] = 1 if pos % 2 == 0 else -1
        for pos, idx in c2_ordered: s_fallback[idx] = 1 if pos % 2 == 0 else -1
        for idx in m_edges: s_fallback[idx] = -1
        best_signs = s_fallback

    return best_signs
