# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    n = 16
    d = 2

    # Temperature for soft-min/soft-max
    T = 0.1

    def soft_min(x, T):
        """Smooth approximation of min(x)"""
        return -T * np.log(np.sum(np.exp(-x / T)))

    def soft_max(x, T):
        """Smooth approximation of max(x)"""
        return T * np.log(np.sum(np.exp(x / T)))

    def objective(x_flat):
        """
        Objective: maximize soft-min distance after normalizing so that soft-max distance = 1.
        Returns negative of what we want to maximize (for minimization).
        """
        x = x_flat.reshape(n, d)
        dists = pdist(x)
        
        if np.allclose(dists, 0):
            return 0.0
        
        D = soft_max(dists, T)
        if D < 1e-10:
            return 0.0
        
        y = x / D
        dists_y = pdist(y)
        m = soft_min(dists_y, T)
        
        return -m

    def true_score(points):
        """Compute the actual score R = (d_min/d_max)^2"""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        return (d_min / d_max) ** 2

    best_points = None
    best_score = -np.inf

    # Diverse initializations
    initializations = []
    
    # 1. Regular polygon (hexadecagon)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. 4x4 grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    initializations.append(np.column_stack([grid_x.flatten(), grid_y.flatten()]))
    
    # 3. Hexagonal lattice subset
    hex_points = []
    for row in range(4):
        for col in range(4):
            x = col * 1.5
            y = row * np.sqrt(3) + (col % 2) * np.sqrt(3) / 2
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    hex_points -= np.mean(hex_points, axis=0)
    D = np.max(pdist(hex_points))
    if D > 0:
        hex_points /= D
    initializations.append(hex_points)
    
    # 4. Random starts
    rng = np.random.RandomState(42)
    for i in range(8):
        x0 = rng.randn(n, d)
        dists = pdist(x0)
        D = np.max(dists)
        if D > 0:
            x0 = x0 / D
        initializations.append(x0)
    
    # 5. Points on two concentric circles
    for r1, r2 in [(0.5, 1.0), (0.3, 0.8), (0.6, 1.0)]:
        angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        initializations.append(np.vstack([p1, p2]))
    
    # Run optimization from each initialization
    for x0 in initializations:
        x0_flat = x0.flatten()
        
        try:
            res = minimize(
                objective,
                x0_flat,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            points = res.x.reshape(n, d)
            score = true_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            continue
    
    # Also try L-BFGS-B with numerical gradient from best point
    if best_points is not None:
        try:
            res2 = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            points2 = res2.x.reshape(n, d)
            score2 = true_score(points2)
            if score2 > best_score:
                best_score = score2
                best_points = points2.copy()
        except Exception:
            pass

    # If all optimizations failed, use regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points


# EVOLVE-BLOCK-END
