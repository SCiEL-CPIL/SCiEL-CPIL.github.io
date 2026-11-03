"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s

    # --- Initial placement: hexagonal lattice ---
    def hex_lattice_init():
        # Try multiple spacings to get at least N points
        best_points = None
        for attempt in range(50):
            # Estimate spacing
            # Area per circle ~ pi*r^2, total area ~ pi, so r ~ 1/sqrt(N)
            # Spacing ~ 2*r
            r_est = 1.0 / np.sqrt(N)
            spacing = 2.0 * r_est * (1.0 + 0.05 * attempt)
            
            k = int(2.0 / spacing) + 3
            rows = []
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    rows.append((x, y))
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            # Keep points well inside
            margin = spacing / 2.0
            mask = radius <= 1.0 - margin
            points = points[mask]
            if len(points) >= N:
                # Sort by distance from center, take first N
                idx = np.argsort(radius[mask], kind="stable")[:N]
                best_points = points[idx]
                break
        
        if best_points is None:
            # Fallback: grid
            n = int(np.sqrt(N)) + 2
            xs = np.linspace(-0.9, 0.9, n)
            ys = np.linspace(-0.9, 0.9, n)
            X, Y = np.meshgrid(xs, ys)
            points = np.column_stack([X.ravel(), Y.ravel()])
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0
            points = points[mask]
            if len(points) >= N:
                best_points = points[:N]
            else:
                # Should not happen
                best_points = np.random.randn(N, 2) * 0.5
        return best_points

    def compute_score(points):
        """Compute the objective: min of pairwise distances/2 and boundary distances."""
        # Pairwise distances
        n = len(points)
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        
        # Boundary distances
        radii = np.sqrt(np.sum(points**2, axis=1))
        min_boundary = np.min(1.0 - radii)
        
        return min(min_pair, min_boundary)

    def objective(x):
        """Negative of the score (we minimize negative score)."""
        points = x.reshape(-1, 2)
        # Penalize if any point is outside
        radii = np.sqrt(np.sum(points**2, axis=1))
        if np.any(radii > 1.0):
            return 1.0 + np.sum(radii[radii > 1.0] - 1.0)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        
        min_boundary = np.min(1.0 - radii)
        
        score = min(min_pair, min_boundary)
        return -score

    def objective_with_grad(x):
        """Objective with gradient for L-BFGS-B."""
        points = x.reshape(-1, 2)
        n = len(points)
        
        radii = np.sqrt(np.sum(points**2, axis=1))
        if np.any(radii > 1.0):
            return 1.0 + np.sum(radii[radii > 1.0] - 1.0), np.zeros_like(x)
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        min_pair = np.min(dist) / 2.0
        min_boundary = np.min(1.0 - radii)
        
        score = min(min_pair, min_boundary)
        
        # Gradient is complex, use finite differences or approximate
        # For now, return zero gradient to let L-BFGS-B use numerical diff
        return -score, np.zeros_like(x)

    # Start with hex lattice
    points = hex_lattice_init()
    x0 = points.ravel()
    current_score = compute_score(points)
    
    # Use L-BFGS-B for optimization
    # We'll do multiple rounds of optimization with decreasing step sizes
    
    # First, let's do a simple gradient-free optimization using Nelder-Mead
    # But N=300 means 600 dimensions, which is too high for Nelder-Mead
    
    # Instead, we'll use a coordinate descent / iterative improvement approach
    # Or use L-BFGS-B with numerical gradients
    
    # Let's try L-BFGS-B
    # But the objective is non-smooth (min of many functions), so gradients are tricky
    
    # Alternative: use a simple iterative approach
    # For each point, move it slightly to improve the minimum distance
    
    best_points = points.copy()
    best_score = current_score
    
    # Iterative improvement: for each point, try small moves in random directions
    # and keep the move if it improves the score
    
    np.random.seed(42)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Try several small moves
        best_move = None
        best_local_score = compute_score(points)
        
        for trial in range(20):
            # Random direction
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
            
            # Small step
            step = 0.001 * (1.0 + 0.1 * trial)
            new_pos = points[idx].copy()
            new_pos += step * direction
            
            # Check if still inside
            if np.linalg.norm(new_pos) > 1.0:
                continue
            
            points[idx] = new_pos
            local_score = compute_score(points)
            
            if local_score > best_local_score:
                best_local_score = local_score
                best_move = new_pos.copy()
            else:
                points[idx] = best_points[idx].copy()
        
        if best_move is not None:
            points[idx] = best_move
            if best_local_score > best_score:
                best_score = best_local_score
                best_points = points.copy()
        else:
            points[idx] = best_points[idx].copy()
    
    # Final polish with L-BFGS-B
    if time.time() < deadline - 10:
        try:
            result = minimize(objective, x0=best_points.ravel(), method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.success:
                new_points = result.x.reshape(-1, 2)
                # Check validity
                radii = np.sqrt(np.sum(new_points**2, axis=1))
                if np.all(radii <= 1.0):
                    new_score = compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
