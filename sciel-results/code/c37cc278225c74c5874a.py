"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        # Compute pairwise distances efficiently
        min_dist = np.inf
        for i in range(N):
            d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            if len(d) > 0:
                min_dist = min(min_dist, np.min(d))
        r = min(min_dist / 2.0, 1.0 - np.max(dists))
        return r
    
    def generate_ring_packing(ring_counts, ring_radii, angular_offsets):
        """Generate points from ring configuration."""
        points = []
        for i, (n_circ, r_ring, offset) in enumerate(zip(ring_counts, ring_radii, angular_offsets)):
            if n_circ == 0:
                continue
            if n_circ == 1:
                if r_ring < 1e-9:
                    points.append([0.0, 0.0])
                else:
                    points.append([r_ring * np.cos(offset), r_ring * np.sin(offset)])
            else:
                angles = offset + np.linspace(0, 2*np.pi, n_circ, endpoint=False)
                for a in angles:
                    points.append([r_ring * np.cos(a), r_ring * np.sin(a)])
        
        points = np.array(points)
        if len(points) != N:
            return None
        return points
    
    def optimize_ring_config(n_rings, initial_radii, initial_counts, initial_offsets):
        """Optimize ring configuration using L-BFGS-B."""
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            radii = params[:n_rings]
            log_counts = params[n_rings:2*n_rings]
            offsets = params[2*n_rings:]
            
            counts = np.maximum(1, np.round(np.exp(log_counts))).astype(int)
            
            if np.sum(counts) != N:
                return 1e10
            
            if np.any(radii < 0) or np.any(radii > 0.999):
                return 1e10
            
            pts = generate_ring_packing(counts, radii, offsets)
            if pts is None:
                return 1e10
            
            score = compute_score(pts)
            if score <= 0:
                return 1e10
            
            return -score
        
        x0 = np.concatenate([initial_radii, np.log(initial_counts.astype(float)), initial_offsets])
        
        bounds = []
        for i in range(n_rings):
            bounds.append((0.0, 0.999))
        for i in range(n_rings):
            bounds.append((np.log(1), np.log(50)))
        for i in range(n_rings):
            bounds.append((-np.pi, np.pi))
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < 1e10:
                radii = result.x[:n_rings]
                log_counts = result.x[n_rings:2*n_rings]
                offsets = result.x[2*n_rings:]
                counts = np.maximum(1, np.round(np.exp(log_counts))).astype(int)
                
                diff = N - np.sum(counts)
                if diff != 0:
                    for i in range(abs(diff)):
                        if diff > 0:
                            counts[i % n_rings] += 1
                        else:
                            counts[i % n_rings] -= 1
                    counts = np.maximum(1, counts)
                
                pts = generate_ring_packing(counts, radii, offsets)
                if pts is not None:
                    return pts, compute_score(pts)
        except Exception:
            pass
        
        return None, -1.0
    
    def epigraph_objective(params):
        """
        Epigraph formulation with auxiliary variable t.
        Maximize t subject to:
        - For all i<j: |c_i - c_j| >= 2t
        - For all i: |c_i| <= 1 - t
        
        We minimize -t + penalty for constraint violations.
        params = [x1, y1, x2, y2, ..., xN, yN, t]
        """
        if time.time() > deadline:
            return 1e10
        
        coords = params[:2*N].reshape(-1, 2)
        t = params[2*N]
        
        # Check validity
        if not np.all(np.isfinite(coords)):
            return 1e10
        
        # Radial constraint violation
        radial_dists = np.linalg.norm(coords, axis=1)
        radial_violation = np.maximum(0, radial_dists - (1.0 - t))
        
        # Pairwise distance constraint violation
        pair_violations = []
        for i in range(N):
            d = np.linalg.norm(coords[i+1:] - coords[i], axis=1)
            if len(d) > 0:
                viol = np.maximum(0, 2*t - d)
                pair_violations.append(np.sum(viol))
        
        pair_violation = sum(pair_violations) if pair_violations else 0.0
        
        # Smooth penalty
        penalty = 1000.0 * (np.sum(radial_violation**2) + pair_violation**2)
        
        return -t + penalty
    
    # Strategy 1: Start with hexagonal lattice as baseline
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    hex_points = None
    for attempt in range(10):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(pts) >= N:
            pts = pts[np.argsort(radius, kind="stable")[:N]]
            hex_points = pts
            break
        spacing *= 0.95
    
    if hex_points is None:
        hex_points = np.zeros((N, 2))
        best_score = -1.0
    else:
        best_score = compute_score(hex_points)
    
    # Strategy 2: Try radial ring configurations
    ring_configs = []
    
    # Configuration 1: 10 rings
    n_rings = 10
    counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54])
    diff = N - np.sum(counts)
    for i in range(abs(diff)):
        counts[i % n_rings] += 1 if diff > 0 else -1
    counts = np.maximum(1, counts)
    radii = np.linspace(0.05, 0.95, n_rings)
    offsets = np.zeros(n_rings)
    ring_configs.append((counts, radii, offsets))
    
    # Configuration 2: 12 rings
    n_rings = 12
    counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66])
    diff = N - np.sum(counts)
    for i in range(abs(diff)):
        counts[i % n_rings] += 1 if diff > 0 else -1
    counts = np.maximum(1, counts)
    radii = np.linspace(0.05, 0.95, n_rings)
    offsets = np.zeros(n_rings)
    ring_configs.append((counts, radii, offsets))
    
    # Configuration 3: 8 rings
    n_rings = 8
    counts = np.array([1, 8, 16, 24, 32, 40, 48, 56])
    diff = N - np.sum(counts)
    for i in range(abs(diff)):
        counts[i % n_rings] += 1 if diff > 0 else -1
    counts = np.maximum(1, counts)
    radii = np.linspace(0.08, 0.95, n_rings)
    offsets = np.zeros(n_rings)
    ring_configs.append((counts, radii, offsets))
    
    # Try each ring configuration
    for counts, radii, offsets in ring_configs:
        if time.time() > deadline - 20:
            break
        
        pts = generate_ring_packing(counts, radii, offsets)
        if pts is not None:
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                hex_points = pts
        
        n_rings = len(counts)
        opt_pts, opt_score = optimize_ring_config(n_rings, radii, counts, offsets)
        if opt_pts is not None and opt_score > best_score:
            best_score = opt_score
            hex_points = opt_pts
    
    # Strategy 3: Epigraph optimization with auxiliary variable t
    if time.time() < deadline - 30 and hex_points is not None:
        # Initial t from current best score
        t0 = max(best_score, 0.01)
        x0 = np.concatenate([hex_points.flatten(), [t0]])
        
        # Bounds: coordinates in [-0.999, 0.999], t in [0, 0.5]
        bounds = [(-0.999, 0.999)] * (2 * N) + [(0.0, 0.5)]
        
        try:
            result = minimize(epigraph_objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < 1e10:
                new_pts = result.x[:2*N].reshape(-1, 2)
                new_t = result.x[2*N]
                new_score = compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    hex_points = new_pts
        except Exception:
            pass
    
    # Strategy 4: Final polish with tighter tolerance
    if time.time() < deadline - 10 and hex_points is not None:
        t0 = max(best_score, 0.01)
        x0 = np.concatenate([hex_points.flatten(), [t0]])
        bounds = [(-0.999, 0.999)] * (2 * N) + [(0.0, 0.5)]
        
        try:
            result = minimize(epigraph_objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < 1e10:
                new_pts = result.x[:2*N].reshape(-1, 2)
                new_score = compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    hex_points = new_pts
        except Exception:
            pass
    
    # Final validation
    if hex_points is None or compute_score(hex_points) <= 0:
        hex_points = np.zeros((N, 2))
        n_per_row = int(np.sqrt(N))
        spacing = 0.03
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            hex_points[i, 0] = (col - n_per_row/2) * spacing
            hex_points[i, 1] = (row - n_per_row/2) * spacing
    
    return hex_points
# EVOLVE-BLOCK-END
