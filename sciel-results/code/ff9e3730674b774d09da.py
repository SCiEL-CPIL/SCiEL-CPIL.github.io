"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return pts / norms
    
    def fibonacci_spiral(phase_offset=0.0):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + phase_offset
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def compute_facet_data(pts):
        """Vectorized computation of facet circumcenters and distances."""
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None
        
        simplices = hull.simplices
        n_facets = len(simplices)
        if n_facets == 0:
            return None
        
        # Extract vertices
        p1 = pts[simplices[:, 0]]
        p2 = pts[simplices[:, 1]]
        p3 = pts[simplices[:, 2]]
        
        # Compute circumcenters vectorized
        u = p2 - p1
        v = p3 - p1
        uu = np.sum(u * u, axis=1)
        vv = np.sum(v * v, axis=1)
        uv = np.sum(u * v, axis=1)
        
        det = uu * vv - uv * uv
        valid = np.abs(det) > 1e-10
        
        if not np.any(valid):
            return None
        
        a = np.zeros(n_facets)
        b = np.zeros(n_facets)
        a[valid] = (0.5 * uu[valid] * vv[valid] - 0.5 * uv[valid] * vv[valid]) / det[valid]
        b[valid] = (0.5 * vv[valid] * uu[valid] - 0.5 * uu[valid] * uv[valid]) / det[valid]
        
        c_planar = p1 + a[:, None] * u + b[:, None] * v
        c_norm = np.linalg.norm(c_planar, axis=1)
        valid &= c_norm > 1e-10
        
        if not np.any(valid):
            return None
        
        c_sph = np.zeros_like(c_planar)
        c_sph[valid] = c_planar[valid] / c_norm[valid, None]
        
        # Check if circumcenter is inside the spherical triangle
        # Using barycentric coordinates: c_sph = alpha*p1 + beta*p2 + gamma*p3
        P = np.stack([p1, p2, p3], axis=1)  # shape (n_facets, 3, 3)
        
        # Solve P @ coeffs = c_sph for each facet
        coeffs = np.zeros((n_facets, 3))
        for i in range(n_facets):
            if not valid[i]:
                continue
            try:
                coeffs[i] = np.linalg.solve(P[i], c_sph[i])
            except np.linalg.LinAlgError:
                coeffs[i, :] = -1.0
        
        is_inside = np.all(coeffs > -1e-6, axis=1) & valid
        
        # Compute distances
        # For inside: distance from circumcenter to vertex
        d_center = np.arccos(np.clip(np.sum(c_sph * p1, axis=1), -1.0, 1.0))
        
        # For outside: midpoint of longest edge
        m12 = (p1 + p2)
        m12_norm = np.linalg.norm(m12, axis=1, keepdims=True)
        m12 = np.where(m12_norm > 1e-10, m12 / np.maximum(m12_norm, 1e-10), p1)
        
        m23 = (p2 + p3)
        m23_norm = np.linalg.norm(m23, axis=1, keepdims=True)
        m23 = np.where(m23_norm > 1e-10, m23 / np.maximum(m23_norm, 1e-10), p2)
        
        m31 = (p3 + p1)
        m31_norm = np.linalg.norm(m31, axis=1, keepdims=True)
        m31 = np.where(m31_norm > 1e-10, m31 / np.maximum(m31_norm, 1e-10), p3)
        
        d_m12 = np.arccos(np.clip(np.sum(m12 * p1, axis=1), -1.0, 1.0))
        d_m23 = np.arccos(np.clip(np.sum(m23 * p2, axis=1), -1.0, 1.0))
        d_m31 = np.arccos(np.clip(np.sum(m31 * p3, axis=1), -1.0, 1.0))
        
        # Determine which edge has the longest distance
        d_edge = np.stack([d_m12, d_m23, d_m31], axis=1)
        max_edge_idx = np.argmax(d_edge, axis=1)
        max_edge_dist = np.max(d_edge, axis=1)
        
        # Combine: use circumcenter if inside, else use edge midpoint
        use_center = is_inside
        y = np.where(use_center[:, None], c_sph, 
                    np.where(max_edge_idx[:, None] == 0, m12,
                          np.where(max_edge_idx[:, None] == 1, m23, m31)))
        d = np.where(use_center, d_center, max_edge_dist)
        
        # Only consider valid facets
        d[~valid] = 0.0
        
        return {
            'simplices': simplices,
            'y': y,
            'd': d,
            'use_center': use_center,
            'max_edge_idx': max_edge_idx,
            'valid': valid
        }
    
    def get_covering_radius_and_grads(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = normalize(pts)
        
        fd = compute_facet_data(pts)
        if fd is None:
            return 1.0, np.zeros_like(x_flat)
        
        d = fd['d']
        valid = fd['valid']
        
        if not np.any(valid):
            return 1.0, np.zeros_like(x_flat)
        
        max_r = np.max(d[valid])
        
        # Identify active facets with tighter tolerance
        tol = 1e-6
        active_mask = valid & (d >= max_r - tol)
        
        if not np.any(active_mask):
            return max_r, np.zeros_like(x_flat)
        
        # Compute gradient
        grads = np.zeros_like(x_flat)
        
        active_indices = np.where(active_mask)[0]
        n_active = len(active_indices)
        
        for idx in active_indices:
            simplex = fd['simplices'][idx]
            y = fd['y'][idx]
            
            if fd['use_center'][idx]:
                # Gradient for circumcenter case: all 3 vertices contribute
                for idx_p in simplex:
                    p = pts[idx_p]
                    dot = np.dot(y, p)
                    denom = np.sqrt(max(1e-10, 1.0 - dot * dot))
                    grads[idx_p * 3: idx_p * 3 + 3] += -y / denom
            else:
                # Gradient for edge midpoint case: both endpoints contribute
                edge_idx = fd['max_edge_idx'][idx]
                if edge_idx == 0:
                    indices = [simplex[0], simplex[1]]
                elif edge_idx == 1:
                    indices = [simplex[1], simplex[2]]
                else:
                    indices = [simplex[2], simplex[0]]
                
                for idx_p in indices:
                    p = pts[idx_p]
                    dot = np.dot(y, p)
                    denom = np.sqrt(max(1e-10, 1.0 - dot * dot))
                    grads[idx_p * 3: idx_p * 3 + 3] += -y / denom
        
        if n_active > 0:
            grads /= n_active
            
        return max_r, grads
    
    def objective_with_grad(x_flat):
        val, grad = get_covering_radius_and_grads(x_flat)
        pts = x_flat.reshape(N, 3)
        pts = normalize(pts)
        
        grad_reshaped = grad.reshape(N, 3)
        dots = np.sum(grad_reshaped * pts, axis=1, keepdims=True)
        grad_tangent = grad_reshaped - dots * pts
        
        return val, grad_tangent.flatten()
    
    def polish(pts, max_iters=50, tol=1e-12):
        """High-precision polish using L-BFGS-B with tight tolerances."""
        x = pts.flatten()
        try:
            res = minimize(
                objective_with_grad,
                x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-10, 'maxls': 20}
            )
            if res.fun < get_covering_radius_and_grads(x)[0]:
                return res.x
        except Exception:
            pass
        return x
    
    # Multi-restart optimization
    best_x = None
    best_val = float('inf')
    
    # Try multiple phase offsets for Fibonacci spiral
    phase_offsets = [0.0, np.pi / 3, 2 * np.pi / 3, np.pi / 5, 2 * np.pi / 5]
    
    for phase in phase_offsets:
        if time.time() > deadline - 30:  # Need at least 30s left
            break
        
        x0 = fibonacci_spiral(phase).flatten()
        val0, _ = get_covering_radius_and_grads(x0)
        
        # Run L-BFGS-B with reasonable iteration budget
        try:
            res = minimize(
                objective_with_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            if val0 < best_val:
                best_val = val0
                best_x = x0
        
        # Polish the best result found so far
        if best_x is not None and time.time() < deadline - 15:
            polished = polish(best_x.reshape(N, 3), max_iters=100, tol=1e-13)
            pol_val, _ = get_covering_radius_and_grads(polished)
            if pol_val < best_val:
                best_val = pol_val
                best_x = polished
    
    if best_x is None:
        # Fallback to original initialization
        best_x = fibonacci_spiral(0.0).flatten()
    
    pts = normalize(best_x.reshape(N, 3))
    return pts
# EVOLVE-BLOCK-END
