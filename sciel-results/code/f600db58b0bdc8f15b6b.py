"""Physics-based relaxation for packing 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Pack 50 equal circles in a unit circle using physics-based relaxation.
    Uses iterative radius expansion with adaptive potential exponent.
    """
    np.random.seed(42)
    
    def score(x):
        """Compute the actual score: the largest common radius."""
        x = x.reshape(N, 2)
        r = np.sqrt(np.sum(x**2, axis=1))
        if np.any(r > 1.0 + 1e-12):
            return 0.0
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e10)
        min_pair_dist = np.min(dist)
        min_wall_dist = np.min(1.0 - r)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    def relax(x, exponent=4.0, max_iter=300, tol=1e-10):
        """
        Relax positions using gradient descent on a smooth energy landscape.
        Returns optimized positions.
        """
        x = x.copy()
        
        def energy_and_grad(x_flat):
            x = x_flat.reshape(N, 2)
            
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            
            iu = np.triu_indices(N, k=1)
            pair_dist = dist[iu]
            pair_dist = np.maximum(pair_dist, 1e-10)
            
            repulsion = np.sum(1.0 / pair_dist**exponent)
            
            r = np.sqrt(np.sum(x**2, axis=1))
            wall_margin = np.maximum(1.0 - r, 1e-10)
            wall = np.sum(1.0 / wall_margin**exponent)
            
            E = repulsion + wall
            
            grad = np.zeros_like(x)
            
            for idx in range(len(iu[0])):
                i, j = iu[0][idx], iu[1][idx]
                d = dist[i, j]
                if d < 1e-10:
                    continue
                coeff = -exponent / d**(exponent + 1)
                diff_ij = x[i] - x[j]
                grad[i] += coeff * diff_ij
                grad[j] -= coeff * diff_ij
            
            for i in range(N):
                ri = r[i]
                if ri < 1e-10:
                    continue
                margin = 1.0 - ri
                if margin < 1e-10:
                    margin = 1e-10
                coeff = exponent / margin**(exponent + 1) * 1.0 / ri
                grad[i] += coeff * x[i]
            
            return E, grad.ravel()
        
        result = minimize(
            energy_and_grad,
            x.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'gtol': tol, 'ftol': tol}
        )
        
        return result.x.reshape(N, 2)
    
    def hexagonal_init(n_points, radius_factor=0.95):
        """Generate hexagonal lattice points inside unit disk."""
        points = []
        spacing = 0.15
        k = int(1.0 / spacing) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                px = (i + 0.5 * (j % 2)) * spacing
                py = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(px**2 + py**2) <= radius_factor:
                    points.append([px, py])
        
        points = np.array(points)
        if len(points) < n_points:
            # Add more points with smaller spacing
            spacing = 0.12
            k = int(1.0 / spacing) + 3
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    px = (i + 0.5 * (j % 2)) * spacing
                    py = j * spacing * np.sqrt(3.0) / 2.0
                    if np.sqrt(px**2 + py**2) <= radius_factor:
                        points = np.vstack([points, [[px, py]]])
        
        if len(points) >= n_points:
            r = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(r)[:n_points]
            return points[idx]
        else:
            # Pad with random points
            remaining = n_points - len(points)
            angles = np.random.uniform(0, 2 * np.pi, remaining)
            radii = np.sqrt(np.random.uniform(0, 0.8, remaining))
            pad = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            return np.vstack([points, pad])
    
    def ring_init(n_points):
        """Generate concentric ring initialization."""
        x_init = np.zeros((n_points, 2))
        # 1 at center
        x_init[0] = [0, 0]
        remaining = n_points - 1
        
        # Distribute on rings
        ring_radii = [0.25, 0.5, 0.75, 0.95]
        ring_counts = [6, 12, 18, 13]  # Total: 1+6+12+18+13 = 50
        
        idx = 1
        for ring_r, count in zip(ring_radii, ring_counts):
            for i in range(count):
                if idx >= n_points:
                    break
                angle = 2 * np.pi * i / count
                x_init[idx] = [ring_r * np.cos(angle), ring_r * np.sin(angle)]
                idx += 1
        
        return x_init
    
    def random_init(n_points):
        """Generate random initialization inside disk."""
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        radii = np.sqrt(np.random.uniform(0, 0.9, n_points))
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    best_x = None
    best_score = -1
    
    # Generate multiple initializations
    inits = []
    inits.append(hexagonal_init(N))
    inits.append(ring_init(N))
    
    for _ in range(4):
        inits.append(random_init(N))
    
    # Phase 1: Relax each initialization with moderate exponent
    for init_x in inits:
        x_relaxed = relax(init_x, exponent=4.0, max_iter=200, tol=1e-8)
        s = score(x_relaxed)
        if s > best_score:
            best_score = s
            best_x = x_relaxed
    
    # Phase 2: Iterative radius expansion
    # Start from best configuration and try to expand
    if best_x is not None:
        current_x = best_x.copy()
        current_score = best_score
        
        for expansion_iter in range(50):
            # Compute current minimum distance
            diff = current_x[:, np.newaxis, :] - current_x[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, 1e10)
            min_pair_dist = np.min(dist)
            r = np.sqrt(np.sum(current_x**2, axis=1))
            min_wall_dist = np.min(1.0 - r)
            current_r = min(min_pair_dist / 2.0, min_wall_dist)
            
            # Try to expand: add a small penalty for being too far from ideal radius
            # Use a softer potential with lower exponent for fine-tuning
            x_new = relax(current_x, exponent=6.0, max_iter=150, tol=1e-10)
            
            s_new = score(x_new)
            if s_new > current_score + 1e-12:
                current_x = x_new
                current_score = s_new
                if s_new > best_score:
                    best_score = s_new
                    best_x = x_new
            else:
                # Try with different exponent
                x_new = relax(current_x, exponent=3.0, max_iter=100, tol=1e-10)
                s_new = score(x_new)
                if s_new > current_score + 1e-12:
                    current_x = x_new
                    current_score = s_new
                    if s_new > best_score:
                        best_score = s_new
                        best_x = x_new
                else:
                    break
    
    # Phase 3: Final polish with very tight tolerance
    if best_x is not None:
        x_final = relax(best_x, exponent=5.0, max_iter=500, tol=1e-12)
        s_final = score(x_final)
        if s_final > best_score:
            best_x = x_final
            best_score = s_final
    
    # Ensure all points are strictly inside
    r = np.sqrt(np.sum(best_x**2, axis=1))
    for i in range(N):
        if r[i] >= 1.0:
            best_x[i] *= 0.9999
    
    return best_x

# EVOLVE-BLOCK-END
