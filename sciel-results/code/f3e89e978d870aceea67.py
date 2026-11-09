"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Identify edge types
    top_cycle = []    # edges in top cycle (0..m-1)
    bottom_cycle = [] # edges in bottom cycle (m..2m-1)
    matching = []     # matching edges (i, i+m)
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle.append(i)
        elif u >= m and v >= m:
            bottom_cycle.append(i)
        else:
            matching.append(i)
    
    def build_signed_adj(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        parent = list(range(2 * n))
        rank = [0] * (2 * n)
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                union(u, v)
                union(u + n, v + n)
            else:
                union(u, v + n)
                union(u + n, v)
        
        root = find(0)
        for i in range(1, 2 * n):
            if find(i) != root:
                return False
        return True
    
    def signs_from_phases(phases):
        """Convert complex phases to edge signs: sign(Re(z_u * conj(z_v)))"""
        signs = np.ones(num_edges, dtype=np.int8)
        for i, (u, v) in enumerate(edges):
            val = np.real(phases[u] * np.conj(phases[v]))
            signs[i] = 1 if val >= 0 else -1
        return signs
    
    def gradient_descent_phases(phases, lr=0.1, steps=500, temperature=1.0):
        """Optimize continuous phases to minimize spectral radius proxy."""
        phases = phases.copy()
        best_radius = float('inf')
        best_phases = phases.copy()
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Compute signs from current phases
            signs = signs_from_phases(phases)
            
            # Check connectivity
            if not is_connected_lift(signs):
                # Perturb and retry
                phases += 0.1 * np.random.randn(n)
                continue
            
            rad = spectral_radius(signs)
            if rad < best_radius:
                best_radius = rad
                best_phases = phases.copy()
            
            # Compute gradient of spectral radius w.r.t. phases
            # Use the fact that d/dtheta of sign(Re(z_u conj(z_v))) 
            # relates to the imaginary part
            # We use a smooth proxy: minimize sum of squared eigenvalues
            # above a threshold, or use the top eigenvalue gradient
            
            A = build_signed_adj(signs)
            eigvals, eigvecs = np.linalg.eigh(A)
            
            # Find eigenvalue with largest absolute value
            idx = np.argmax(np.abs(eigvals))
            lam = eigvals[idx]
            v = eigvecs[:, idx]
            
            # Gradient of |lambda_max| w.r.t. A entries
            # d|lambda|/dA_ij = sign(lambda) * v_i * v_j
            grad_A = np.sign(lam) * np.outer(v, v)
            
            # Map gradient from A to phases
            # A[u,v] = sign(Re(z_u conj(z_v)))
            # d(Re(z_u conj(z_v)))/d(theta_u) = -Im(z_u conj(z_v))
            # d(Re(z_u conj(z_v)))/d(theta_v) = Im(z_u conj(z_v))
            
            grad_theta = np.zeros(n)
            for i, (u, v) in enumerate(edges):
                g = grad_A[u, v]
                # The sign function is non-differentiable, but we use the 
                # underlying continuous quantity
                phase_diff = np.angle(phases[u] * np.conj(phases[v]))
                # Gradient of Re(z_u conj(z_v)) w.r.t. theta_u = -sin(phase_diff)
                # Gradient of Re(z_u conj(z_v)) w.r.t. theta_v = sin(phase_diff)
                sin_val = np.sin(phase_diff)
                grad_theta[u] -= g * sin_val
                grad_theta[v] += g * sin_val
            
            # Apply gradient descent with adaptive learning rate
            grad_norm = np.linalg.norm(grad_theta)
            if grad_norm > 1e-10:
                phases -= lr * grad_theta / grad_norm
                # Keep phases in [0, 2*pi)
                phases = np.mod(phases, 2 * np.pi)
            
            # Occasionally add noise to escape local minima
            if step % 50 == 0:
                phases += 0.05 * np.random.randn(n)
                phases = np.mod(phases, 2 * np.pi)
        
        return best_phases, best_radius
    
    def local_search(signs, max_steps=3000):
        """Flip-based local search to polish the result."""
        curr = signs.copy()
        if not is_connected_lift(curr):
            return None, float('inf')
        
        curr_rad = spectral_radius(curr)
        best_rad = curr_rad
        best_signs = curr.copy()
        
        T = 0.3
        cooling = 0.998
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            idx = np.random.randint(0, num_edges)
            new_signs = curr.copy()
            new_signs[idx] *= -1
            
            if not is_connected_lift(new_signs):
                continue
            
            new_rad = spectral_radius(new_signs)
            delta = new_rad - curr_rad
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                curr = new_signs
                curr_rad = new_rad
                if curr_rad < best_rad:
                    best_rad = curr_rad
                    best_signs = curr.copy()
            
            T *= cooling
        
        return best_signs, best_rad
    
    rng = np.random.default_rng(42)
    
    best_radius = float('inf')
    best_signs = np.ones(num_edges, dtype=np.int8)
    
    # Phase synchronization approach: multiple restarts with different initial phases
    for restart in range(8):
        if time.time() > deadline:
            break
        
        # Initialize phases
        if restart == 0:
            # Structured initialization: alternating phases on each cycle
            phases = np.zeros(n)
            for i in range(m):
                phases[i] = i * np.pi / m  # top cycle
                phases[m + i] = i * np.pi / m + np.pi / 2  # bottom cycle offset
        else:
            phases = rng.random(n) * 2 * np.pi
        
        # Run gradient descent on phases
        optimized_phases, opt_radius = gradient_descent_phases(phases, lr=0.05, steps=300)
        
        # Convert to signs
        signs = signs_from_phases(optimized_phases)
        
        # Ensure connectivity
        if not is_connected_lift(signs):
            # Try to fix
            for _ in range(100):
                if is_connected_lift(signs):
                    break
                idx = rng.integers(0, num_edges)
                signs[idx] *= -1
        
        if not is_connected_lift(signs):
            continue
        
        rad = spectral_radius(signs)
        if rad < best_radius:
            best_radius = rad
            best_signs = signs.copy()
        
        # Polish with local search
        if time.time() < deadline - 5:
            polished, polished_rad = local_search(signs, max_steps=1000)
            if polished is not None and polished_rad < best_radius:
                best_radius = polished_rad
                best_signs = polished.copy()
    
    # Final fallback
    if not is_connected_lift(best_signs):
        for i in range(num_edges):
            signs = np.full(num_edges, -1, dtype=np.int8)
            signs[i] = 1
            if is_connected_lift(signs):
                best_signs = signs
                break
    
    return best_signs
# EVOLVE-BLOCK-END
