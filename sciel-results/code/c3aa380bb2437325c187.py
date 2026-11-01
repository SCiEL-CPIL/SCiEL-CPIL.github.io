"""Program for packing 50 equal circles in a unit square.

Approach: continuous rigid-body assembly via sparse contact graph.
We start from a good initial guess (grid perturbation + known packings),
build a sparse distance-constraint system, and iteratively adjust positions
to maximize the minimum pairwise distance and boundary clearance.
The key idea is to treat the packing as a contact network where only
nearby circles interact, and solve the resulting sparse linear system
to push apart overlapping circles and pull them away from walls.
"""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize


def _build_contact_graph(centers, threshold):
    """Build sparse contact graph: pairs of circles within threshold distance."""
    n = len(centers)
    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < threshold:
                rows.append(i)
                cols.append(j)
                data.append(d)
    if not rows:
        return csr_matrix((0, n), dtype=float), np.array([]), np.array([])
    mat = csr_matrix((data, (rows, cols)), shape=(len(rows), n))
    return mat, np.array(rows), np.array(cols)


def _objective_and_grad(x, n, r_target):
    """Objective: minimize sum of violations (overlaps + wall penetration).
    x is flattened centers (2n values). Returns (f, grad).
    """
    centers = x.reshape(n, 2)
    f = 0.0
    grad = np.zeros(2 * n)
    
    # Wall constraints: each center must be at least r from each wall
    # Square is [-0.5, 0.5]^2, so distance to wall is min(0.5 - |x|, 0.5 - |y|)
    for i in range(n):
        for k in range(2):
            coord = centers[i, k]
            dist_wall = 0.5 - abs(coord)
            if dist_wall < r_target:
                viol = r_target - dist_wall
                f += viol ** 2
                # d/dcoord of viol^2 where viol = r - (0.5 - |coord|)
                # d viol / d coord = -sign(coord) (for coord != 0)
                sign = 1.0 if coord > 0 else (-1.0 if coord < 0 else 0.0)
                grad[2 * i + k] += 2.0 * viol * sign
    
    # Pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            dvec = centers[i] - centers[j]
            d = np.linalg.norm(dvec)
            if d < 1e-12:
                dvec = np.array([1e-8, 0.0])
                d = 1e-8
            min_dist = 2.0 * r_target
            if d < min_dist:
                viol = min_dist - d
                f += viol ** 2
                # d viol / d centers[i] = -dvec/d, d viol / d centers[j] = dvec/d
                grad[2 * i:2 * i + 2] += 2.0 * viol * (-dvec / d)
                grad[2 * j:2 * j + 2] += 2.0 * viol * (dvec / d)
    
    return f, grad


def _max_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    min_wall = np.inf
    for i in range(n):
        for k in range(2):
            dist = 0.5 - abs(centers[i, k])
            if dist < min_wall:
                min_wall = dist
    return min(min_pair, 2.0 * min_wall) / 2.0


def _initial_grid():
    """Create a perturbed grid initial guess."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, size=centers.shape)
    # Clip to stay inside
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _known_packing_seed():
    """Try to use a known-good packing structure as seed.
    Based on Boll et al. 2000 packing for 50 circles.
    We approximate with a hexagonal-ish pattern.
    """
    # Hexagonal packing pattern
    centers = []
    # Use rows with alternating offsets
    row_spacing = 0.1427  # approx 2*0.07137
    col_spacing = 0.1427
    y_offset = 0.5 - row_spacing / 2
    
    # Generate hexagonal grid points
    rows = []
    y = -0.5 + row_spacing / 2
    while y < 0.5:
        rows.append(y)
        y += row_spacing * np.sqrt(3) / 2
    
    for idx, y in enumerate(rows):
        x_start = -0.5 + col_spacing / 2
        if idx % 2 == 1:
            x_start += col_spacing / 2
        x = x_start
        while x < 0.5:
            centers.append([x, y])
            x += col_spacing
    
    centers = np.array(centers[:N])
    if len(centers) < N:
        # Pad with grid
        grid = _initial_grid()
        centers = np.vstack([centers, grid[len(centers):]])
    
    # Center the packing
    centers -= centers.mean(axis=0)
    # Scale to fit
    max_extent = np.max(np.abs(centers))
    if max_extent > 0.4:
        centers *= 0.4 / max_extent
    
    return centers


def construct():
    # Start with multiple initial guesses and pick the best
    initial_guesses = [
        _initial_grid(),
        _known_packing_seed(),
    ]
    
    best_centers = None
    best_radius = 0.0
    
    for init_centers in initial_guesses:
        # First, compute current radius
        r_current = _max_radius(init_centers)
        
        # Iterative refinement: increase target radius and solve
        # Use a sequence of increasing target radii
        r_target = r_current * 0.9
        centers = init_centers.copy()
        
        for iteration in range(50):
            # Solve for positions that satisfy constraints at r_target
            # Use L-BFGS-B style optimization on the violation objective
            x = centers.flatten()
            
            # Use scipy minimize with gradient
            result = minimize(
                _objective_and_grad,
                x,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            new_centers = result.x.reshape(N, 2)
            
            # Check if solution is valid
            r_new = _max_radius(new_centers)
            
            if r_new > best_radius:
                best_radius = r_new
                best_centers = new_centers.copy()
            
            # Increase target radius for next iteration
            r_target = min(r_new * 1.02, 0.1)
            
            if result.fun < 1e-10:
                # Constraints satisfied, can increase target
                r_target = r_new * 1.01
            else:
                # Not fully satisfied, try a smaller increase
                r_target = (r_new + r_target) / 2 * 1.005
            
            centers = new_centers
        
        # Final polish with direct radius maximization
        # Try to further improve by optimizing directly
        x = best_centers.flatten()
        result = minimize(
            lambda x: -_max_radius(x.reshape(N, 2)),
            x,
            method='Nelder-Mead',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.fun < -best_radius:
            best_radius = -result.fun
            best_centers = result.x.reshape(N, 2)
    
    # Ensure valid output
    if best_centers is None:
        best_centers = _initial_grid()
    
    # Final validation: ensure all points are strictly inside
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
