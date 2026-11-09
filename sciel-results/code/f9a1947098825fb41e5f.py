# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Returns a valid packing of 10 squares in the unit square maximizing the sum of side lengths.
    
    Uses gradient-based optimization starting from the known analytical construction:
    - 2 large squares of side 0.5 at (0.25, 0.25) and (0.75, 0.75).
    - 8 small squares of side 0.25 filling the remaining top-left and bottom-right 0.5x0.5 quadrants.
    
    Total sum of sides = 2 * 0.5 + 8 * 0.25 = 1.0 + 2.0 = 3.0.
    Baseline sum = 2.5.
    Score = 3.0 / 2.5 = 1.2.
    """
    start_time = time.time()
    deadline = 580.0  # Safety margin before 600s limit
    
    n = 10
    tol = 1e-7
    
    # Initial analytical solution
    centers_init = np.array([
        [0.25, 0.25],
        [0.75, 0.75],
        [0.125, 0.625],
        [0.375, 0.625],
        [0.125, 0.875],
        [0.375, 0.875],
        [0.625, 0.125],
        [0.875, 0.125],
        [0.625, 0.375],
        [0.875, 0.375]
    ], dtype=np.float64)
    sides_init = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    
    # Best valid solution found
    best_centers = centers_init.copy()
    best_sides = sides_init.copy()
    best_sum = np.sum(sides_init)
    
    def is_valid(centers, sides):
        """Check if the packing is valid."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if not np.all(sides > 0):
            return False
        # Check inside unit square with tolerance
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # Check no overlap
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j])/2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True
    
    def pack_to_vec(centers, sides):
        """Pack centers and sides into a single vector."""
        return np.concatenate([centers.flatten(), sides])
    
    def unpack_from_vec(vec):
        """Unpack vector into centers and sides."""
        centers = vec[:2*n].reshape(n, 2)
        sides = vec[2*n:]
        return centers, sides
    
    def penalty(centers, sides, beta=100.0):
        """Penalty function for constraints."""
        pen = 0.0
        # Boundary constraints
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Left boundary
            d = -tol - (cx - s/2)
            if d > 0:
                pen += beta * d**2
            # Right boundary
            d = (cx + s/2) - (1 + tol)
            if d > 0:
                pen += beta * d**2
            # Bottom boundary
            d = -tol - (cy - s/2)
            if d > 0:
                pen += beta * d**2
            # Top boundary
            d = (cy + s/2) - (1 + tol)
            if d > 0:
                pen += beta * d**2
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j])/2 - tol
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                # If overlapping, add penalty
                if dx < t and dy < t:
                    # Penalty proportional to how much they overlap
                    overlap_x = t - dx
                    overlap_y = t - dy
                    pen += beta * (overlap_x**2 + overlap_y**2)
        return pen
    
    def objective(vec):
        """Objective: minimize negative sum of sides + penalty."""
        centers, sides = unpack_from_vec(vec)
        # Ensure sides are positive
        sides = np.maximum(sides, 1e-6)
        sum_sides = np.sum(sides)
        pen = penalty(centers, sides)
        return -sum_sides + pen
    
    def objective_grad(vec):
        """Gradient of the objective function."""
        centers, sides = unpack_from_vec(vec)
        sides = np.maximum(sides, 1e-6)
        
        grad = np.zeros_like(vec)
        
        # Gradient of -sum(sides)
        grad[2*n:] = -1.0
        
        # Gradient of penalty
        # Boundary constraints
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Left boundary: d = -tol - (cx - s/2)
            if cx - s/2 < -tol:
                d = -tol - (cx - s/2)
                grad[2*i] += 2*beta*d*(-1)  # d/dc_x
                grad[2*i+1] += 0
                grad[2*n+i] += 2*beta*d*(0.5)  # d/ds
            # Right boundary: d = (cx + s/2) - (1 + tol)
            if cx + s/2 > 1 + tol:
                d = (cx + s/2) - (1 + tol)
                grad[2*i] += 2*beta*d*(1)
                grad[2*i+1] += 0
                grad[2*n+i] += 2*beta*d*(0.5)
            # Bottom boundary: d = -tol - (cy - s/2)
            if cy - s/2 < -tol:
                d = -tol - (cy - s/2)
                grad[2*i] += 0
                grad[2*i+1] += 2*beta*d*(-1)
                grad[2*n+i] += 2*beta*d*(0.5)
            # Top boundary: d = (cy + s/2) - (1 + tol)
            if cy + s/2 > 1 + tol:
                d = (cy + s/2) - (1 + tol)
                grad[2*i] += 0
                grad[2*i+1] += 2*beta*d*(1)
                grad[2*n+i] += 2*beta*d*(0.5)
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j])/2 - tol
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                adx = abs(dx)
                ady = abs(dy)
                if adx < t and ady < t:
                    overlap_x = t - adx
                    overlap_y = t - ady
                    # Gradient with respect to positions
                    if dx > 0:
                        grad[2*i] += 2*beta*overlap_x*(1)
                        grad[2*j] += 2*beta*overlap_x*(-1)
                    else:
                        grad[2*i] += 2*beta*overlap_x*(-1)
                        grad[2*j] += 2*beta*overlap_x*(1)
                    if dy > 0:
                        grad[2*i+1] += 2*beta*overlap_y*(1)
                        grad[2*j+1] += 2*beta*overlap_y*(-1)
                    else:
                        grad[2*i+1] += 2*beta*overlap_y*(-1)
                        grad[2*j+1] += 2*beta*overlap_y*(1)
                    # Gradient with respect to sides
                    grad[2*n+i] += 2*beta*(overlap_x + overlap_y)*(0.5)
                    grad[2*n+j] += 2*beta*(overlap_x + overlap_y)*(0.5)
        
        return grad
    
    # Try optimization with different initial perturbations
    np.random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            x0 = pack_to_vec(centers_init, sides_init)
        else:
            # Add small random perturbation
            centers_pert = centers_init + np.random.normal(0, 0.01, size=(n, 2))
            sides_pert = sides_init + np.random.normal(0, 0.005, size=n)
            sides_pert = np.maximum(sides_pert, 0.05)
            x0 = pack_to_vec(centers_pert, sides_pert)
        
        # Run optimization
        try:
            result = minimize(
                objective,
                x0,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success:
                centers_opt, sides_opt = unpack_from_vec(result.x)
                sides_opt = np.maximum(sides_opt, 1e-6)
                
                if is_valid(centers_opt, sides_opt):
                    sum_opt = np.sum(sides_opt)
                    if sum_opt > best_sum:
                        best_centers = centers_opt
                        best_sides = sides_opt
                        best_sum = sum_opt
                        
        except Exception as e:
            pass
        
        if time.time() > deadline:
            break
    
    # Final validation
    if not is_valid(best_centers, best_sides):
        # Fall back to initial solution
        best_centers = centers_init
        best_sides = sides_init
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
