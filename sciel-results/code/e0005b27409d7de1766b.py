"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place N=50 equal circles in unit circle to maximize common radius.
    
    Uses a multi-start local optimization approach:
    1. Generate multiple initial configurations (hexagonal grid, random, ring-based)
    2. For each, optimize the radius by maximizing the minimum constraint
       (min of pairwise distances/2 and distance to boundary)
    3. Return the best configuration found.
    """
    best_radius = 0.0
    best_points = None
    
    # Objective: maximize the minimum of all constraints
    # We'll minimize the negative of this
    
    def objective(params):
        """
        params: flat array of 2*N coordinates
        Returns: negative of the minimum radius constraint
        """
        pts = params.reshape(N, 2)
        
        # Distance from each center to boundary
        dist_to_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        
        # Pairwise distances
        # Compute all pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        # The radius is limited by half the min pairwise distance and min distance to boundary
        r = min(min_pairwise / 2.0, np.min(dist_to_boundary))
        
        return -r
    
    def gradient(params):
        """
        Numerical gradient for the objective.
        """
        eps = 1e-8
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus)
            
            params_minus = params.copy()
            params_minus[i] -= eps
            f_minus = objective(params_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    # Generate initial configurations
    def hexagonal_initial():
        """Hexagonal lattice initial configuration."""
        # Try different spacings
        best = None
        for spacing in np.linspace(0.25, 0.45, 20):
            k = int(1.0 / spacing) + 3
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            
            if len(points) >= N:
                # Select N points closest to center for better packing
                idx = np.argsort(radius)[:N]
                pts = points[idx]
                # Check validity
                r = _compute_radius(pts)
                if r > 0 and (best is None or r > best[0]):
                    best = (r, pts)
        
        if best:
            return best[1]
        return None
    
    def random_initial(seed=42):
        """Random initial configuration inside the unit circle."""
        rng = np.random.RandomState(seed)
        while True:
            pts = rng.uniform(-1, 1, size=(N, 2))
            mask = np.linalg.norm(pts, axis=1) <= 0.9
            if np.sum(mask) >= N:
                pts = pts[mask][:N]
                r = _compute_radius(pts)
                if r > 0.05:
                    return pts
            # Retry with different seed
            seed += 1
    
    def ring_initial():
        """Ring-based initial configuration."""
        pts = np.zeros((N, 2))
        # Center point
        pts[0] = [0, 0]
        # First ring
        n1 = 6
        for i in range(n1):
            angle = 2 * np.pi * i / n1
            pts[i + 1] = [0.35 * np.cos(angle), 0.35 * np.sin(angle)]
        # Second ring
        n2 = 12
        offset = 1 + n1
        for i in range(n2):
            angle = 2 * np.pi * i / n2 + np.pi / n2
            pts[i + offset] = [0.65 * np.cos(angle), 0.65 * np.sin(angle)]
        # Third ring
        n3 = N - 1 - n1 - n2
        offset = 1 + n1 + n2
        for i in range(n3):
            angle = 2 * np.pi * i / n3
            pts[i + offset] = [0.9 * np.cos(angle), 0.9 * np.sin(angle)]
        
        return pts
    
    def _compute_radius(pts):
        """Compute the maximum valid radius for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Distance from each center to boundary
        dist_to_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        if np.any(dist_to_boundary < 0):
            return 0.0
        
        # Pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        r = min(min_pairwise / 2.0, np.min(dist_to_boundary))
        return max(r, 0.0)
    
    # Try different initializations
    initializations = []
    
    # Hexagonal
    hex_pts = hexagonal_initial()
    if hex_pts is not None:
        initializations.append(hex_pts)
    
    # Random with different seeds
    for seed in [42, 123, 456, 789, 101, 202]:
        try:
            rand_pts = random_initial(seed)
            if rand_pts is not None:
                initializations.append(rand_pts)
        except:
            pass
    
    # Ring-based
    try:
        ring_pts = ring_initial()
        initializations.append(ring_pts)
    except:
        pass
    
    # Optimize each initial configuration
    for init_pts in initializations:
        if init_pts is None:
            continue
        
        # Flatten
        x0 = init_pts.flatten()
        
        # Use L-BFGS-B with numerical gradient
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            if result.success or result.fun < 0:
                pts = result.x.reshape(N, 2)
                r = _compute_radius(pts)
                if r > best_radius:
                    best_radius = r
                    best_points = pts
        except Exception as e:
            pass
    
    # If no good result, fallback to hexagonal
    if best_points is None:
        best_points = hexagonal_initial()
        if best_points is None:
            # Last resort: simple grid
            best_points = np.zeros((N, 2))
            for i in range(N):
                angle = 2 * np.pi * i / N
                best_points[i] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    
    return best_points

# EVOLVE-BLOCK-END
