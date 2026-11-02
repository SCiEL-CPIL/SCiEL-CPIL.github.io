"""Circle packing in unit square n=32: maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 32
DEADLINE = 120.0  # wall-clock limit
SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def _pack_circles():
    """
    Optimize circle packing via gradient-based optimization with penalty method.
    Uses a smooth penalty for constraints to handle the non-smooth min-distance.
    """
    start_time = time.time()
    
    # --- Initialization: multiple strategies ---
    def grid_init():
        """Grid-based initialization with slight randomization."""
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        radii = []
        for i in range(side):
            for j in range(side):
                if len(positions) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Add small random perturbation
                x += np.random.uniform(-0.01, 0.01)
                y += np.random.uniform(-0.01, 0.01)
                positions.append([x, y])
                r = 0.5 / side * 0.95
                radii.append(r)
        # Pad if needed
        while len(positions) < N:
            positions.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
            radii.append(0.05)
        return np.array(positions[:N]), np.array(radii[:N])

    def hex_init():
        """Hexagonal close-packing inspired initialization."""
        positions = []
        radii = []
        # Hexagonal lattice
        dx = 0.18
        dy = dx * np.sqrt(3) / 2
        row = 0
        while len(positions) < N:
            offset = dx / 2 if row % 2 else 0
            col = 0
            while len(positions) < N:
                x = 0.1 + offset + col * dx
                y = 0.1 + row * dy
                if 0 < x < 1 and 0 < y < 1:
                    positions.append([x, y])
                    radii.append(dx / 2 * 0.95)
                col += 1
            row += 1
        # Pad if needed
        while len(positions) < N:
            positions.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
            radii.append(0.05)
        return np.array(positions[:N]), np.array(radii[:N])

    def random_init():
        """Random initialization."""
        positions = np.random.uniform(0.1, 0.9, size=(N, 2))
        radii = np.full(N, 0.08)
        return positions, radii

    inits = [grid_init, hex_init, random_init]
    
    best_positions = None
    best_radii = None
    best_score = -np.inf

    def objective_and_constraints(pos_rad):
        """
        pos_rad: array of shape (N, 3) where columns are (x, y, r)
        Returns: (objective_value, constraint_violation)
        """
        positions = pos_rad[:, :2]
        radii = pos_rad[:, 2]
        
        # Objective: maximize sum of radii
        obj = np.sum(radii)
        
        # Constraints:
        # 1. Circles inside unit square: x >= r, y >= r, x <= 1-r, y <= 1-r
        # 2. Non-overlapping: dist(i,j) >= r_i + r_j
        
        # Boundary violations
        bound_viol = 0.0
        for i in range(N):
            x, y, r = positions[i, 0], positions[i, 1], radii[i]
            if r < 0:
                bound_viol += abs(r)
            if x < r:
                bound_viol += r - x
            if y < r:
                bound_viol += r - y
            if x > 1 - r:
                bound_viol += x - (1 - r)
            if y > 1 - r:
                bound_viol += y - (1 - r)
        
        # Overlap violations
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap_viol += min_dist - dist
        
        total_viol = bound_viol + overlap_viol
        return obj, total_viol

    def penalty_objective(pos_rad, penalty_weight):
        """Penalized objective: maximize sum(r) - penalty * violation"""
        positions = pos_rad[:, :2]
        radii = pos_rad[:, 2]
        
        # Clip radii to be non-negative for evaluation
        radii_clipped = np.maximum(radii, 0)
        
        obj = np.sum(radii_clipped)
        
        # Boundary violations
        bound_viol = 0.0
        for i in range(N):
            x, y, r = positions[i, 0], positions[i, 1], radii_clipped[i]
            if x < r:
                bound_viol += r - x
            if y < r:
                bound_viol += r - y
            if x > 1 - r:
                bound_viol += x - (1 - r)
            if y > 1 - r:
                bound_viol += y - (1 - r)
        
        # Overlap violations
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_dist = radii_clipped[i] + radii_clipped[j]
                if dist < min_dist:
                    overlap_viol += min_dist - dist
        
        total_viol = bound_viol + overlap_viol
        return -(obj - penalty_weight * total_viol)

    def penalty_objective_grad(pos_rad, penalty_weight):
        """Gradient of penalized objective."""
        positions = pos_rad[:, :2]
        radii = pos_rad[:, 2]
        radii_clipped = np.maximum(radii, 0)
        
        grad = np.zeros_like(pos_rad)
        
        # Gradient of sum(r) w.r.t. radii
        for i in range(N):
            grad[i, 2] = -1.0  # We're minimizing -(sum(r) - penalty*viol), so d/d r_i = -1
        
        # Gradient of boundary violations
        for i in range(N):
            x, y, r = positions[i, 0], positions[i, 1], radii_clipped[i]
            # x < r: violation = r - x, d/dx = -1, d/dr = 1
            if x < r:
                grad[i, 0] += penalty_weight * (-1)
                grad[i, 2] += penalty_weight * (1)
            # y < r
            if y < r:
                grad[i, 1] += penalty_weight * (-1)
                grad[i, 2] += penalty_weight * (1)
            # x > 1-r: violation = x - (1-r) = x - 1 + r, d/dx = 1, d/dr = 1
            if x > 1 - r:
                grad[i, 0] += penalty_weight * (1)
                grad[i, 2] += penalty_weight * (1)
            # y > 1-r
            if y > 1 - r:
                grad[i, 1] += penalty_weight * (1)
                grad[i, 2] += penalty_weight * (1)
        
        # Gradient of overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                min_dist = radii_clipped[i] + radii_clipped[j]
                if dist < min_dist and dist > 1e-10:
                    viol = min_dist - dist
                    # d(viol)/d(pos_i) = -diff/dist, d(viol)/d(r_i) = 1
                    grad[i, 0] += penalty_weight * (-diff[0] / dist)
                    grad[i, 1] += penalty_weight * (-diff[1] / dist)
                    grad[i, 2] += penalty_weight * (1)
                    # d(viol)/d(pos_j) = diff/dist, d(viol)/d(r_j) = 1
                    grad[j, 0] += penalty_weight * (diff[0] / dist)
                    grad[j, 1] += penalty_weight * (diff[1] / dist)
                    grad[j, 2] += penalty_weight * (1)
        
        return grad

    # Optimization loop with increasing penalty
    for init_func in inits:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        positions, radii = init_func()
        x0 = np.column_stack([positions, radii])
        
        # Progressive penalty increase
        penalty_weights = [10, 100, 1000, 5000]
        current_x = x0.copy()
        
        for pw in penalty_weights:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Use L-BFGS-B with bounds
            bounds = []
            for i in range(N):
                bounds.append((0, 1))  # x
                bounds.append((0, 1))  # y
                bounds.append((0, 0.5))  # r
        
            try:
                result = minimize(
                    penalty_objective,
                    current_x,
                    args=(pw,),
                    method='L-BFGS-B',
                    jac=penalty_objective_grad,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_x = result.x
            except Exception:
                pass
        
        # Evaluate final solution
        final_obj, final_viol = objective_and_constraints(current_x)
        if final_viol < 1e-6:
            score = final_obj
        else:
            # Penalize invalid solutions heavily
            score = final_obj - 100 * final_viol
        
        if score > best_score:
            best_score = score
            best_positions = current_x[:, :2].copy()
            best_radii = current_x[:, 2].copy()

    # Final polish: try to improve with tighter tolerance
    if best_positions is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        x0 = np.column_stack([best_positions, best_radii])
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
        
        try:
            result = minimize(
                penalty_objective,
                x0,
                args=(10000,),
                method='L-BFGS-B',
                jac=penalty_objective_grad,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            final_obj, final_viol = objective_and_constraints(result.x)
            if final_viol < 1e-6:
                score = final_obj
                if score > best_score:
                    best_score = score
                    best_positions = result.x[:, :2].copy()
                    best_radii = result.x[:, 2].copy()
        except Exception:
            pass

    # Ensure valid output
    if best_positions is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        while len(cells) < N:
            cells.append([0.5, 0.5, 0.01])
        return np.array(cells[:N])
    
    # Clip to valid range
    best_radii = np.maximum(best_radii, 0)
    for i in range(N):
        best_radii[i] = min(best_radii[i], min(best_positions[i, 0], best_positions[i, 1],
                                                1 - best_positions[i, 0], 1 - best_positions[i, 1]))
    
    return np.column_stack([best_positions, best_radii])


def construct():
    """Construct the circle packing configuration."""
    np.random.seed(42)
    result = _pack_circles()
    return result
