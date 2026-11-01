"""Circle packing in unit square n=32."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

def _hex_initialization(n):
    """Generate a hexagonal packing initialization for n circles."""
    # Hexagonal lattice spacing
    # For hexagonal packing, the ratio of vertical to horizontal spacing is sqrt(3)/2
    # Let's place circles in a hexagonal pattern
    # Start with a reasonable estimate
    # For n circles in a unit square, average area per circle is 1/n
    # Radius ~ sqrt(1/(4*pi*n)) ~ 0.05 for n=32
    
    # Generate hexagonal positions
    # Use a triangular lattice
    # The packing density of hexagonal packing is pi/(2*sqrt(3)) ~ 0.9069
    # For n circles, total area = n * pi * r^2
    # n * pi * r^2 <= 0.9069 * 1 (approximate, ignoring boundary effects)
    # r <= sqrt(0.9069 / (n * pi))
    
    # Let's try a grid-based hexagonal arrangement
    # Number of columns
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Hexagonal offset
    # Horizontal spacing
    # Vertical spacing
    # For hexagonal packing with radius r:
    # Horizontal distance between centers in same row: 2r
    # Vertical distance between rows: 2r * sqrt(3)/2 = r*sqrt(3)
    # Odd rows are offset by r horizontally
    
    # Let's estimate r
    # If we have 'rows' rows and 'cols' columns:
    # Width: cols * 2r + r (for offset) <= 1  =>  r <= 1/(2*cols + 1)
    # Height: (rows-1) * r*sqrt(3) + 2r <= 1  =>  r <= 1/((rows-1)*sqrt(3) + 2)
    
    # Try different arrangements
    best_r = 0
    best_positions = None
    
    for cols_try in range(1, n + 1):
        rows_try = int(np.ceil(n / cols_try))
        
        # Calculate max radius for this arrangement
        # Width constraint: cols_try * 2r + r <= 1  (if offset needed)
        # Actually, for hexagonal:
        # Even rows: centers at x = r, 3r, 5r, ...
        # Odd rows: centers at x = 2r, 4r, 6r, ...
        # So max x in even row: (2*cols_try - 1) * r
        # Max x in odd row: 2 * cols_try * r
        # We need both <= 1 - r (so that x + r <= 1)
        # (2*cols_try - 1) * r + r <= 1  =>  2*cols_try * r <= 1  =>  r <= 1/(2*cols_try)
        # 2*cols_try * r + r <= 1  =>  (2*cols_try + 1) * r <= 1  =>  r <= 1/(2*cols_try + 1)
        
        # Height constraint:
        # (rows_try - 1) * r * sqrt(3) + 2r <= 1
        # r * ((rows_try - 1) * sqrt(3) + 2) <= 1
        # r <= 1 / ((rows_try - 1) * sqrt(3) + 2)
        
        r_width = 1.0 / (2 * cols_try + 1)
        r_height = 1.0 / ((rows_try - 1) * np.sqrt(3) + 2)
        r = min(r_width, r_height)
        
        if r > best_r:
            best_r = r
            # Generate positions
            positions = []
            for row in range(rows_try):
                for col in range(cols_try):
                    if len(positions) >= n:
                        break
                    if row % 2 == 0:
                        x = (2 * col + 1) * r
                    else:
                        x = (2 * col + 2) * r
                    y = r + row * r * np.sqrt(3)
                    positions.append((x, y))
                    if len(positions) >= n:
                        break
            best_positions = positions
    
    if best_positions is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return cells[:n]
    
    return best_positions


def _random_initialization(n, seed=None):
    """Generate random initialization with push-apart."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Start with random positions
    positions = []
    for _ in range(n):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        positions.append([x, y])
    
    # Push apart overlapping circles
    # Estimate initial radius
    r_est = 0.5 / np.sqrt(n)
    
    for iteration in range(200):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = 2 * r_est
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    push = (min_dist - dist) / 2
                    nx = dx / dist
                    ny = dy / dist
                    positions[i][0] -= push * nx
                    positions[i][1] -= push * ny
                    positions[j][0] += push * nx
                    positions[j][1] += push * ny
                    moved = True
                elif dist <= 1e-10:
                    # Randomly separate
                    positions[i][0] += random.uniform(-0.01, 0.01)
                    positions[i][1] += random.uniform(-0.01, 0.01)
                    moved = True
        if not moved:
            break
    
    # Clamp to [0, 1]
    for i in range(n):
        positions[i][0] = np.clip(positions[i][0], 0, 1)
        positions[i][1] = np.clip(positions[i][1], 0, 1)
    
    return positions


def _softmin(values, k=50):
    """Soft minimum using log-sum-exp."""
    # softmin(x) = -logsumexp(-x)
    neg_values = -values
    max_val = np.max(neg_values)
    if max_val > 500:
        # Avoid overflow
        shifted = neg_values - max_val
        result = -max_val - np.log(np.sum(np.exp(shifted)))
    else:
        result = -np.log(np.sum(np.exp(neg_values)))
    return result


def _clearance(positions, radii):
    """Calculate minimum clearance (distance between centers minus sum of radii)."""
    n = len(positions)
    positions = np.array(positions)
    radii = np.array(radii)
    
    min_clearance = np.inf
    
    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            clearance = dist - radii[i] - radii[j]
            if clearance < min_clearance:
                min_clearance = clearance
    
    # Boundary
    for i in range(n):
        clearance = min(positions[i, 0] - radii[i],
                        1 - positions[i, 0] - radii[i],
                        positions[i, 1] - radii[i],
                        1 - positions[i, 1] - radii[i])
        if clearance < min_clearance:
            min_clearance = clearance
    
    return min_clearance


def _objective(x, n, penalty_weight=100.0):
    """Objective function: maximize sum of radii with penalty for violations."""
    # x contains: 2*n positions + n radii
    positions = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Penalty for negative radii
    penalty = penalty_weight * np.sum(np.clip(-radii, 0, None) ** 2)
    
    # Penalty for boundary violations
    for i in range(n):
        boundary_violation = max(0, radii[i] - positions[i, 0]) + \
                              max(0, radii[i] - (1 - positions[i, 0])) + \
                              max(0, radii[i] - positions[i, 1]) + \
                              max(0, radii[i] - (1 - positions[i, 1]))
        penalty += penalty_weight * boundary_violation ** 2
    
    # Penalty for overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = max(0, radii[i] + radii[j] - dist)
            penalty += penalty_weight * overlap ** 2
    
    return -np.sum(radii) + penalty


def _objective_grad(x, n, penalty_weight=100.0):
    """Gradient of the objective function."""
    positions = x[:2*n].reshape(n, 2)
    radii = x[2*n:].copy()
    
    grad = np.zeros_like(x)
    
    # d/dradii: -1 + penalty terms
    grad[2*n:] = -1.0
    
    # Boundary penalty gradient
    for i in range(n):
        # d/dx_i
        if positions[i, 0] < radii[i]:
            grad[i*2] += 2 * penalty_weight * (radii[i] - positions[i, 0])
        if positions[i, 0] > 1 - radii[i]:
            grad[i*2] -= 2 * penalty_weight * (positions[i, 0] + radii[i] - 1)
        
        # d/dy_i
        if positions[i, 1] < radii[i]:
            grad[i*2+1] += 2 * penalty_weight * (radii[i] - positions[i, 1])
        if positions[i, 1] > 1 - radii[i]:
            grad[i*2+1] -= 2 * penalty_weight * (positions[i, 1] + radii[i] - 1)
        
        # d/dr_i from boundary
        boundary_viol = 0
        if positions[i, 0] < radii[i]:
            boundary_viol += (radii[i] - positions[i, 0])
        if positions[i, 0] > 1 - radii[i]:
            boundary_viol += (positions[i, 0] + radii[i] - 1)
        if positions[i, 1] < radii[i]:
            boundary_viol += (radii[i] - positions[i, 1])
        if positions[i, 1] > 1 - radii[i]:
            boundary_viol += (positions[i, 1] + radii[i] - 1)
        grad[2*n+i] += 2 * penalty_weight * boundary_viol
    
    # Overlap penalty gradient
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 1e-10:
                overlap = max(0, radii[i] + radii[j] - dist)
                if overlap > 0:
                    factor = 2 * penalty_weight * overlap / dist
                    grad[i*2] += factor * dx
                    grad[i*2+1] += factor * dy
                    grad[j*2] -= factor * dx
                    grad[j*2+1] -= factor * dy
                    grad[2*n+i] += 2 * penalty_weight * overlap
                    grad[2*n+j] += 2 * penalty_weight * overlap
    
    # Negative radii penalty
    for i in range(n):
        if radii[i] < 0:
            grad[2*n+i] += 2 * penalty_weight * (-radii[i])
    
    return grad


def _polish(positions, radii, time_limit=30.0):
    """High-precision polish pass."""
    n = len(positions)
    
    # Start from current solution
    x0 = np.concatenate([np.array(positions).flatten(), np.array(radii)])
    
    start_time = time.time()
    
    # Use L-BFGS-B with tighter tolerance
    result = minimize(
        _objective,
        x0,
        jac=_objective_grad,
        args=(n, 500.0),
        method='L-BFGS-B',
        bounds=[(0, 1)] * (2*n) + [(0, 0.5)] * n,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
    )
    
    if time.time() - start_time > time_limit:
        return positions, radii
    
    x_opt = result.x
    positions_opt = x_opt[:2*n].reshape(n, 2)
    radii_opt = x_opt[2*n:]
    
    # Ensure validity
    valid = True
    for i in range(n):
        if radii_opt[i] < 0 or positions_opt[i, 0] < 0 or positions_opt[i, 0] > 1 or \
           positions_opt[i, 1] < 0 or positions_opt[i, 1] > 1:
            valid = False
            break
        if positions_opt[i, 0] + radii_opt[i] > 1 + 1e-6 or positions_opt[i, 1] + radii_opt[i] > 1 + 1e-6:
            valid = False
            break
        if positions_opt[i, 0] - radii_opt[i] < -1e-6 or positions_opt[i, 1] - radii_opt[i] < -1e-6:
            valid = False
            break
    
    if valid:
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions_opt[j, 0] - positions_opt[i, 0]
                dy = positions_opt[j, 1] - positions_opt[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii_opt[i] + radii_opt[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    if valid:
        return positions_opt, radii_opt
    
    return positions, radii


def construct():
    n = N
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Try multiple initializations
    initializations = []
    
    # Hexagonal
    hex_pos = _hex_initialization(n)
    initializations.append(hex_pos)
    
    # Grid
    side = int(np.ceil(np.sqrt(n)))
    grid_pos = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)][:n]
    initializations.append(grid_pos)
    
    # Random with different seeds
    for seed in [42, 123, 456, 789, 101]:
        if time.time() > deadline - 30:
            break
        rand_pos = _random_initialization(n, seed=seed)
        initializations.append(rand_pos)
    
    best_score = -np.inf
    best_result = None
    
    for init_idx, positions in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        positions = np.array(positions, dtype=float)
        
        # Estimate initial radius
        # Find minimum distance between centers
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist
        
        # Also consider boundary
        for i in range(n):
            min_dist = min(min_dist, 2 * positions[i, 0], 2 * (1 - positions[i, 0]),
                          2 * positions[i, 1], 2 * (1 - positions[i, 1]))
        
        r_init = min_dist / 2 * 0.9  # Slightly smaller to be safe
        
        # Initial radii
        radii = np.full(n, r_init)
        
        # Build initial x
        x0 = np.concatenate([positions.flatten(), radii])
        
        # Optimize
        try:
            result = minimize(
                _objective,
                x0,
                jac=_objective_grad,
                args=(n, 100.0),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2*n) + [(0, 0.5)] * n,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
        except Exception:
            continue
        
        x_opt = result.x
        positions_opt = x_opt[:2*n].reshape(n, 2)
        radii_opt = x_opt[2*n:]
        
        # Validate
        valid = True
        for i in range(n):
            if radii_opt[i] < -1e-6:
                valid = False
                break
            if positions_opt[i, 0] < -1e-6 or positions_opt[i, 0] > 1 + 1e-6:
                valid = False
                break
            if positions_opt[i, 1] < -1e-6 or positions_opt[i, 1] > 1 + 1e-6:
                valid = False
                break
            if positions_opt[i, 0] + radii_opt[i] > 1 + 1e-5:
                valid = False
                break
            if positions_opt[i, 1] + radii_opt[i] > 1 + 1e-5:
                valid = False
                break
            if positions_opt[i, 0] - radii_opt[i] < -1e-5:
                valid = False
                break
            if positions_opt[i, 1] - radii_opt[i] < -1e-5:
                valid = False
                break
        
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions_opt[j, 0] - positions_opt[i, 0]
                    dy = positions_opt[j, 1] - positions_opt[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii_opt[i] + radii_opt[j] - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(radii_opt)
            if score > best_score:
                best_score = score
                best_result = (positions_opt.copy(), radii_opt.copy())
        
        # Polish if time allows
        if best_result is not None and time.time() < deadline - 5:
            polished_pos, polished_rad = _polish(best_result[0], best_result[1], time_limit=10.0)
            polished_score = np.sum(polished_rad)
            if polished_score > best_score:
                best_score = polished_score
                best_result = (polished_pos, polished_rad)
    
    # Return best result
    if best_result is not None:
        positions, radii = best_result
        result = np.zeros((n, 3))
        result[:, 0] = positions[:, 0]
        result[:, 1] = positions[:, 1]
        result[:, 2] = radii
        return result
    
    # Fallback to grid
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:n])
# EVOLVE-BLOCK-END
