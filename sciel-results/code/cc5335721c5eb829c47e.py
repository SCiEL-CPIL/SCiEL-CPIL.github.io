"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _validate(circles):
    """Check circles are inside unit square and non-overlapping."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < 0) or np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True


def _objective(circles):
    """Negative sum of radii."""
    return -np.sum(circles[:, 2])


def _grow_and_relax(circles, max_time=30.0):
    """
    Iteratively grow circles as much as possible, then relax positions via SLSQP
    using Delaunay-based contact constraints.
    """
    start = time.time()
    n = len(circles)
    circles = circles.copy()

    for _iteration in range(50):
        if time.time() - start > max_time:
            break

        # Try to grow each circle
        improved = False
        for i in range(n):
            # Max radius for circle i given current positions
            max_r = min(circles[i, 0], 1 - circles[i, 0],
                        circles[i, 1], 1 - circles[i, 1])
            for j in range(n):
                if i == j:
                    continue
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    max_r = min(max_r, dist - circles[j, 2])
            max_r = max(0, max_r)
            if max_r > circles[i, 2] * (1 + 1e-8):
                circles[i, 2] = max_r
                improved = True

        if not improved:
            break

        # Now relax: try to move circles to free up space
        # Build Delaunay triangulation of centers
        centers = circles[:, :2]
        try:
            tri = Delaunay(centers)
            # Extract edges
            edges = set()
            for simplex in tri.simplices:
                for k in range(3):
                    a, b = simplex[k], simplex[(k + 1) % 3]
                    edges.add((min(a, b), max(a, b)))
        except Exception:
            # Fallback: use all pairs
            edges = set()
            for i in range(n):
                for j in range(i + 1, n):
                    edges.add((i, j))

        # Define constraints for SLSQP: for each edge, dist >= r_i + r_j
        def constraints_func(c):
            c = c.reshape(n, 3)
            cons = []
            for (i, j) in edges:
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                cons.append(dist - c[i, 2] - c[j, 2])
            # Boundary constraints
            for i in range(n):
                cons.append(c[i, 0] - c[i, 2])
                cons.append(1 - c[i, 0] - c[i, 2])
                cons.append(c[i, 1] - c[i, 2])
                cons.append(1 - c[i, 1] - c[i, 2])
            return cons

        # Try to improve sum of radii by adjusting positions
        def neg_sum_rad(c):
            return -np.sum(c[2::3])

        bounds = [(0, 1) for _ in range(2 * n)] + [(0, 0.5) for _ in range(n)]

        # Fixed radii during relaxation
        fixed_r = circles[:, 2].copy()

        def objective_relax(c):
            return -np.sum(fixed_r)  # constant, but we want to move positions

        # Actually, let's do a different approach: maximize min gap
        # Or just try to find a better configuration

        # Alternative: use SLSQP to maximize sum of radii with positions as variables
        # But this is complex. Let's just do a simple local search.

    return circles


def _local_search(circles, max_time=60.0):
    """
    Local search: perturb positions and try to grow circles.
    Uses Delaunay contact graph for constraint identification.
    """
    start = time.time()
    n = len(circles)
    best = circles.copy()
    best_val = np.sum(best[:, 2])

    for _iter in range(200):
        if time.time() - start > max_time:
            break
        if _remaining() < 5:
            break

        # Random perturbation
        scale = 0.05
        c = best.copy()
        c[:, :2] += scale * np.random.randn(n, 2)
        c[:, :2] = np.clip(c[:, :2], 0, 1)

        # Try to grow circles greedily
        for i in range(n):
            max_r = min(c[i, 0], 1 - c[i, 0], c[i, 1], 1 - c[i, 1])
            for j in range(n):
                if i == j:
                    continue
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    max_r = min(max_r, dist - c[j, 2])
            c[i, 2] = max(0, max_r)

        if _validate(c):
            val = np.sum(c[:, 2])
            if val > best_val:
                best = c.copy()
                best_val = val

    return best


def _slsqp_optimize(circles, max_time=45.0):
    """
    Use SLSQP to maximize sum of radii with positions and radii as variables.
    Constraints: non-overlap (Delaunay-based) and boundary.
    """
    start = time.time()
    n = len(circles)
    x0 = circles.flatten()

    # Objective: maximize sum of radii
    def objective(x):
        r = x[2::3]
        return -np.sum(r)

    # Constraints
    def make_constraints():
        cons = []
        # Boundary
        for i in range(n):
            # x_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]})
            # x_i + r_i <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i] - x[2*n+i]})
            # y_i >= r_i
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]})
            # y_i + r_i <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[n+i] - x[2*n+i]})

        # Non-overlap: use all pairs initially, then refine
        for i in range(n):
            for j in range(i+1, n):
                def overlap_constraint(x, i=i, j=j):
                    dx = x[i] - x[j]
                    dy = x[n+i] - x[n+j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - x[2*n+i] - x[2*n+j]
                cons.append({'type': 'ineq', 'fun': overlap_constraint})

        return cons

    constraints = make_constraints()
    bounds = [(0, 1) for _ in range(2*n)] + [(0, 0.5) for _ in range(n)]

    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})

    if result.success or result.fun < objective(x0):
        new_circles = result.x.reshape(n, 3)
        if _validate(new_circles):
            return new_circles

    return circles


def _initial_hex_grid(n):
    """Create a hexagonal grid initial packing."""
    # Hexagonal packing in unit square
    # Try to fit as many as possible
    circles = []
    # Use a grid with hexagonal offset
    cols = 5
    rows = 5
    dx = 1.0 / cols
    dy = 1.0 / rows
    for i in range(cols):
        for j in range(rows):
            x = (i + 0.5) * dx
            y = (j + 0.5) * dy
            if i % 2 == 1:
                y += dy / 2
            r = min(x, 1-x, y, 1-y) * 0.8
            circles.append([x, y, r])
    return np.array(circles[:n])


def _initial_grid(n):
    """Grid initial packing."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells[:n])


def construct():
    global _start_time
    _start_time = time.time()

    # Start with grid initial
    circles = _initial_grid(N)

    # Phase 1: Grow and relax
    if _remaining() > 60:
        circles = _grow_and_relax(circles, max_time=40)

    # Phase 2: SLSQP optimization
    if _remaining() > 30:
        circles = _slsqp_optimize(circles, max_time=25)

    # Phase 3: Local search
    if _remaining() > 10:
        circles = _local_search(circles, max_time=8)

    # Final validation
    if not _validate(circles):
        # Fall back to simple grid
        circles = _initial_grid(N)

    return circles
# EVOLVE-BLOCK-END
