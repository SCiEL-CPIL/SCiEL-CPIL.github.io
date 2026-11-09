import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    def compute_points_from_spherical(sph_params):
        """Convert 28D spherical params to 14x3 Cartesian points on unit sphere."""
        theta = sph_params[:n]
        phi = sph_params[n:]
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        points = np.column_stack((x, y, z))
        return points
    
    def objective(sph_params):
        """Minimize negative of (d_min/d_max)^2."""
        points = compute_points_from_spherical(sph_params)
        
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-15:
            return 0.0
        
        ratio = d_min / d_max
        return -ratio * ratio
    
    def evaluate_score(points):
        """Compute the actual score for a set of points."""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    
    best_score = -1.0
    best_points = None
    
    # Multi-start with diverse initializations
    np.random.seed(42)
    
    # Start 1: Random
    # Start 2: Fibonacci sphere
    # Start 3: Perturbed known good configurations
    
    for start in range(30):
        if start == 0:
            # Pure random
            theta = np.random.uniform(0, 2 * np.pi, n)
            phi = np.random.uniform(0, np.pi, n)
        elif start == 1:
            # Fibonacci sphere - well-distributed
            golden = (1 + np.sqrt(5)) / 2
            i = np.arange(n)
            phi = np.arccos(1 - 2 * (i + 0.5) / n)
            theta = 2 * np.pi * i / golden
            # Add small random perturbation
            theta += np.random.uniform(-0.1, 0.1, n)
            phi += np.random.uniform(-0.05, 0.05, n)
            phi = np.clip(phi, 0.01, np.pi - 0.01)
        elif start == 2:
            # Two antipodal clusters with perturbation
            base_theta = np.random.uniform(0, 2 * np.pi)
            base_phi = np.random.uniform(0.3, np.pi - 0.3)
            theta = np.tile(base_theta, 7) + np.random.uniform(-0.5, 0.5, n)
            phi = np.tile(base_phi, 7) + np.random.uniform(-0.3, 0.3, n)
            # Second cluster antipodal
            theta[7:] += np.pi
            phi[7:] = np.pi - phi[7:]
            phi = np.clip(phi, 0.01, np.pi - 0.01)
        elif start == 3:
            # Regular tetrahedron-like arrangement with extra points
            # 4 vertices of tetrahedron, then 10 more distributed
            tetra_theta = np.array([0, 2*np.pi/3, 4*np.pi/3, 0])
            tetra_phi = np.array([np.arccos(1/3), np.arccos(1/3), np.arccos(1/3), np.arccos(-1/3)])
            theta = np.concatenate([tetra_theta, np.random.uniform(0, 2*np.pi, 10)])
            phi = np.concatenate([tetra_phi, np.random.uniform(0.3, np.pi-0.3, 10)])
            theta += np.random.uniform(-0.2, 0.2, n)
            phi += np.random.uniform(-0.1, 0.1, n)
            phi = np.clip(phi, 0.01, np.pi - 0.01)
        else:
            # Perturb previous best if we have one
            if best_points is not None:
                z = best_points[:, 2]
                y = best_points[:, 1]
                x = best_points[:, 0]
                z = np.clip(z, -1.0, 1.0)
                phi = np.arccos(z)
                theta = np.arctan2(y, x)
                theta += np.random.uniform(-0.3, 0.3, n)
                phi += np.random.uniform(-0.15, 0.15, n)
                phi = np.clip(phi, 0.01, np.pi - 0.01)
            else:
                theta = np.random.uniform(0, 2 * np.pi, n)
                phi = np.random.uniform(0.1, np.pi - 0.1, n)
        
        init_sph = np.concatenate([theta, phi])
        
        try:
            # Nelder-Mead optimization
            res = minimize(objective, init_sph, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True})
            
            if res.fun < 0:  # res.fun is negative score
                final_sph = res.x
                final_points = compute_points_from_spherical(final_sph)
                final_score = evaluate_score(final_points)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points
                    
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: random points
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    
    return best_points
