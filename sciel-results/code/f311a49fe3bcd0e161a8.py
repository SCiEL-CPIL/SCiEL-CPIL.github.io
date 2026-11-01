"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32


def construct():
    """
    Pack 32 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Returns array of shape (n, 3): rows (x, y, radius).
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    # Initial guess: hexagonal-like packing
    def _hexagonal_init():
        """Create initial positions based on hexagonal packing."""
        # Use a grid with slight perturbation
        n_cols = 6
        n_rows = 6
        positions = []
        for i in range(n_cols):
            for j in range(n_rows):
                x = (i + 0.5) / n_cols
                y = (j + 0.5) / n_rows
                # Add slight hexagonal offset
                if j % 2 == 1:
                    x += 0.25 / n_cols
                positions.append([x, y])
        
        # Take first N positions
        positions = positions[:N]
        
        # Initial radii: estimate based on spacing
        # For a grid of 6x6, spacing is about 1/6
        # Radius should be less than half the spacing
        base_radius = 0.5 / max(n_cols, n_rows) * 0.95
        radii = [base_radius] * N
        
        return np.array(positions), np.array(radii)

    def _grid_init():
        """Create initial positions based on square grid."""
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y])
        
        positions = positions[:N]
        base_radius = 0.5 / side * 0.95
        radii = [base_radius] * N
        
        return np.array(positions), np.array(radii)

    def _random_init():
        """Create random initial positions."""
        positions = np.random.uniform(0.1, 0.9, (N, 2))
        radii = np.full(N, 0.05)
        return positions, radii

    def _objective(params):
        """
        Objective function: minimize negative sum of radii plus penalty for violations.
        params: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
        """
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10  # timeout penalty
        
        # Unpack
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Penalty for circles going outside [0,1]^2
        boundary_penalty = 0.0
        for i in range(N):
            if x[i] < r[i]:
                boundary_penalty += (r[i] - x[i]) ** 2
            if x[i] > 1.0 - r[i]:
                boundary_penalty += (x[i] - (1.0 - r[i])) ** 2
            if y[i] < r[i]:
                boundary_penalty += (r[i] - y[i]) ** 2
            if y[i] > 1.0 - r[i]:
                boundary_penalty += (y[i] - (1.0 - r[i])) ** 2
        
        # Penalty for overlapping circles
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist) ** 2
        
        # Maximize sum of radii (minimize negative)
        neg_sum_r = -np.sum(r)
        
        # Weighted penalty
        penalty_weight = 500.0
        return neg_sum_r + penalty_weight * (boundary_penalty + overlap_penalty)

    def _run_optimization(x0, y0, r0, maxiter=500):
        """Run L-BFGS-B optimization from given initial point."""
        params = np.concatenate([x0, y0, r0])
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.0, 0.5))  # r
        
        result = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        # Extract results
        x = result.x[0::3]
        y = result.x[1::3]
        r = result.x[2::3]
        
        # Clip to valid range
        r = np.clip(r, 0.0, 0.5)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        return x, y, r, result.fun

    def _validate(x, y, r):
        """Check if circles are valid (no overlaps, inside square)."""
        # Check boundaries
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] > 1.0 - r[i] + 1e-9:
                return False
            if y[i] < r[i] - 1e-9 or y[i] > 1.0 - r[i] + 1e-9:
                return False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        
        return True

    def _fix_violations(x, y, r, max_iters=100):
        """Iteratively fix violations by reducing radii or moving circles."""
        for _ in range(max_iters):
            if time.time() - start_time > deadline:
                break
            
            violated = False
            
            # Fix boundary violations
            for i in range(N):
                if x[i] < r[i]:
                    r[i] = max(0.0, x[i] * 0.99)
                    violated = True
                if x[i] > 1.0 - r[i]:
                    r[i] = max(0.0, (1.0 - x[i]) * 0.99)
                    violated = True
                if y[i] < r[i]:
                    r[i] = max(0.0, y[i] * 0.99)
                    violated = True
                if y[i] > 1.0 - r[i]:
                    r[i] = max(0.0, (1.0 - y[i]) * 0.99)
                    violated = True
            
            # Fix overlap violations
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        # Reduce radii proportionally
                        excess = min_dist - dist
                        r[i] -= excess * 0.5
                        r[j] -= excess * 0.5
                        r[i] = max(0.0, r[i])
                        r[j] = max(0.0, r[j])
                        violated = True
            
            if not violated:
                break
        
        return x, y, r

    # Try multiple initializations
    best_score = -np.inf
    best_result = None

    initializers = [_hexagonal_init, _grid_init, _random_init]
    
    for init_func in initializers:
        if time.time() - start_time > deadline:
            break
        
        x0, y0, r0 = init_func()
        
        # Run optimization
        x, y, r, fun = _run_optimization(x0, y0, r0, maxiter=300)
        
        # Fix any remaining violations
        x, y, r = _fix_violations(x, y, r)
        
        # Calculate score
        score = np.sum(r)
        
        if score > best_score:
            best_score = score
            best_result = (x, y, r)
    
    # If we have time, try to improve with more iterations from best result
    if best_result is not None and time.time() - start_time < deadline - 10:
        x, y, r = best_result
        x, y, r, fun = _run_optimization(x, y, r, maxiter=200)
        x, y, r = _fix_violations(x, y, r)
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_result = (x, y, r)

    # Final fallback if no valid result
    if best_result is None:
        x, y, r = _grid_init()
        best_result = (x, y, r)

    x, y, r = best_result
    
    # Build result array
    result = np.column_stack([x, y, r])
    
    return result
