"""Improved circle packing for N=300 in unit circle using multi-restart gradient optimization."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check all points are inside unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_pair, min_boundary)

    def generate_hex_init(spacing_scale=1.0):
        """Generate a hexagonal lattice configuration scaled to fit."""
        best_pts = None
        best_score = 0.0
        
        for scale in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]:
            d = 0.106 * scale  # base spacing
            # Generate hexagonal grid
            points = []
            n_rows = int(2.0 / (d * np.sqrt(3) / 2)) + 5
            n_cols = int(2.0 / d) + 5
            
            for row in range(-n_rows//2, n_rows//2 + 1):
                for col in range(-n_cols//2, n_cols//2 + 1):
                    x = col * d
                    y = row * d * np.sqrt(3) / 2
                    if row % 2 != 0:
                        x += d / 2
                    points.append([x, y])
            
            points = np.array(points)
            
            # Filter to unit circle
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii < 0.98
            candidates = points[mask]
            
            if len(candidates) < N:
                continue
            
            # Greedy selection: pick points that maximize minimum distance
            selected = []
            remaining = candidates.copy()
            
            # Find point closest to origin
            dists_origin = np.hypot(remaining[:, 0], remaining[:, 1])
            idx = np.argmin(dists_origin)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
            while len(selected) < N and len(remaining) > 0:
                selected_arr = np.array(selected)
                diff = remaining[:, np.newaxis, :] - selected_arr[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                min_dists = np.min(dists, axis=1)
                idx = np.argmax(min_dists)
                selected.append(remaining[idx])
                remaining = np.delete(remaining, idx, axis=0)
                
                if time.time() > deadline - 5:
                    break
            
            if len(selected) < N:
                continue
            
            pts = np.array(selected[:N])
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
        
        if best_pts is None:
            # Fallback: simple grid
            pts = np.random.uniform(-0.95, 0.95, (N, 2))
            radii = np.hypot(pts[:, 0], pts[:, 1])
            outside = radii > 0.95
            pts[outside] = pts[outside] * (0.95 / radii[outside, np.newaxis])
            best_pts = pts
            best_score = compute_score(pts)
        
        return best_pts, best_score

    def objective_and_grad(flat_pts):
        """Objective function and gradient for scipy minimize (negative score)."""
        pts = flat_pts.reshape(N, 2)
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - radii
        
        # Current score
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(boundary_dists)
        score = min(min_pair, min_boundary)
        
        # Gradient: we want to maximize score, so minimize -score
        # But score is min of many terms, so gradient is tricky
        # Use a smooth approximation: penalize violations
        r_target = score + 0.001  # target slightly above current
        
        # Pairwise repulsion gradient
        forces = np.zeros_like(pts)
        overlap = 2.0 * r_target - dists
        overlap[overlap < 0] = 0.0
        dists_safe = np.where(dists > 1e-10, dists, 1.0)
        dir_vecs = diff / dists_safe[:, :, np.newaxis]
        weighted_dirs = dir_vecs * overlap[:, :, np.newaxis]
        forces += np.sum(weighted_dirs, axis=1)
        
        # Boundary gradient
        boundary_violation = radii - (1.0 - r_target)
        boundary_violation[boundary_violation < 0] = 0.0
        radii_safe = np.where(radii > 1e-10, radii, 1.0)
        boundary_dirs = -pts / radii_safe[:, np.newaxis]
        forces += boundary_dirs * boundary_violation[:, np.newaxis]
        
        # Negative gradient (we want to maximize)
        grad = -forces.flatten()
        
        return -score, grad

    def optimize_with_scipy(pts, max_time=30.0):
        """Use scipy.optimize.minimize with L-BFGS-B."""
        start = time.time()
        flat_pts = pts.flatten()
        
        # Bounds: keep points inside unit circle (approximately)
        bounds = [(-1.0, 1.0)] * (N * 2)
        
        try:
            result = minimize(
                objective_and_grad,
                flat_pts,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'maxfun': 100000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -compute_score(pts):
                new_pts = result.x.reshape(N, 2)
                # Project inside unit circle
                radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                outside = radii > 1.0
                if np.any(outside):
                    new_pts[outside] = new_pts[outside] / radii[outside, np.newaxis]
                return new_pts
        except Exception:
            pass
        
        return pts

    def local_search(pts, max_iters=5000, deadline_check=None):
        """
        Local search: perturb one point at a time, keep if better.
        Focus on points that are close to others or close to boundary.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        temp = 0.01
        
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Find the most constrained point (closest to another or boundary)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dists = np.min(dists, axis=1)
            
            radii = np.hypot(pts[:, 0], pts[:, 1])
            boundary_dists = 1.0 - radii
            
            # Combined constraint score
            constraint_score = np.minimum(min_pair_dists / 2.0, boundary_dists)
            
            # Bias toward more constrained points
            weights = 1.0 / (constraint_score + 0.01)
            weights /= np.sum(weights)
            
            idx = np.random.choice(N, p=weights)
            orig_pt = pts[idx].copy()
            
            # Perturb
            pts[idx] += np.random.randn(2) * temp
            
            # Project inside
            r = np.hypot(pts[idx, 0], pts[idx, 1])
            if r > 1.0:
                pts[idx] = pts[idx] / r
            
            new_score = compute_score(pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
                temp *= 0.995
            else:
                pts[idx] = orig_pt
                temp *= 0.9999
            
            # Reset temperature if too small
            if temp < 0.002:
                temp = 0.008
        
        return best_pts, best_score

    def force_relaxation(pts, n_iters=1000, deadline_check=None):
        """Vectorized force-directed relaxation."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        for it in range(n_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Compute pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            current_score = min(np.min(dists) / 2.0, np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1])))
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            
            r_target = best_score * 1.01 + 0.0005
            
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion
            overlap = 2.0 * r_target - dists
            overlap[overlap < 0] = 0.0
            dists_safe = np.where(dists > 1e-10, dists, 1.0)
            dir_vecs = diff / dists_safe[:, :, np.newaxis]
            forces += np.sum(dir_vecs * overlap[:, :, np.newaxis], axis=1)
            
            # Boundary constraint
            radii = np.hypot(pts[:, 0], pts[:, 1])
            boundary_violation = radii - (1.0 - r_target)
            boundary_violation[boundary_violation < 0] = 0.0
            radii_safe = np.where(radii > 1e-10, radii, 1.0)
            forces += (-pts / radii_safe[:, np.newaxis]) * boundary_violation[:, np.newaxis]
            
            # Adaptive step size
            step = 0.002 / (1.0 + it / 500.0)
            pts += forces * step
            
            # Project inside unit circle
            new_radii = np.hypot(pts[:, 0], pts[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
        
        return best_pts, best_score

    # Main execution
    # 1. Initialize with hexagonal lattice
    pts, init_score = generate_hex_init()
    
    best_pts = pts.copy()
    best_score = init_score
    
    # 2. Force relaxation
    remaining_time = deadline - time.time()
    if remaining_time > 40:
        pts, score = force_relaxation(pts, n_iters=2000, deadline_check=deadline - 40)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # 3. Local search
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        pts, score = local_search(best_pts, max_iters=3000, deadline_check=deadline - 10)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # 4. Second round of force relaxation with different init
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        # Try a different initialization: concentric rings
        ring_pts = []
        # Center
        ring_pts.append([0.0, 0.0])
        
        # Rings
        for ring in range(1, 20):
            r_ring = 0.05 * ring
            if r_ring > 0.95:
                break
            n_in_ring = int(2 * np.pi * r_ring / 0.106)
            n_in_ring = max(n_in_ring, 6)
            for i in range(n_in_ring):
                angle = 2 * np.pi * i / n_in_ring
                ring_pts.append([r_ring * np.cos(angle), r_ring * np.sin(angle)])
        
        ring_pts = np.array(ring_pts)
        if len(ring_pts) >= N:
            # Select N points with max min distance
            selected = [ring_pts[0]]
            remaining = np.delete(ring_pts, 0, axis=0)
            while len(selected) < N and len(remaining) > 0:
                sel_arr = np.array(selected)
                diff = remaining[:, np.newaxis, :] - sel_arr[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                min_dists = np.min(dists, axis=1)
                idx = np.argmax(min_dists)
                selected.append(remaining[idx])
                remaining = np.delete(remaining, idx, axis=0)
            
            if len(selected) >= N:
                ring_pts = np.array(selected[:N])
                ring_score = compute_score(ring_pts)
                
                if ring_score > best_score * 0.95:
                    pts, score = force_relaxation(ring_pts, n_iters=1500, deadline_check=deadline - 15)
                    if score > best_score:
                        best_score = score
                        best_pts = pts
                    pts, score = local_search(best_pts, max_iters=2000, deadline_check=deadline - 5)
                    if score > best_score:
                        best_score = score
                        best_pts = pts
    
    # 5. Final local search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        pts, score = local_search(best_pts, max_iters=1000, deadline_check=deadline - 1)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Final validation
    final_score = compute_score(best_pts)
    if final_score > best_score:
        best_score = final_score
    
    # Ensure validity
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
    
    return best_pts
# EVOLVE-BLOCK-END
