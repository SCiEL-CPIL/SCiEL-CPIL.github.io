"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _sum_radii(circles):
    return np.sum(circles[:, 2])


def _is_valid(circles, tol=1e-7):
    # Check inside unit square
    if np.any(circles[:, 0] - circles[:, 2] < -tol) or np.any(circles[:, 1] - circles[:, 2] < -tol):
        return False
    if np.any(circles[:, 0] + circles[:, 2] > 1 + tol) or np.any(circles[:, 1] + circles[:, 2] > 1 + tol):
        return False
    # Check pairwise non-overlap
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i, 2] + circles[j, 2] - tol:
                return False
    return True


def _penalty(circles, lam=1e4):
    """Penalty for invalidity."""
    n = len(circles)
    pen = 0.0
    # Wall violations
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0: pen += (x - r) ** 2
        if y - r < 0: pen += (y - r) ** 2
        if x + r > 1: pen += (x + r - 1) ** 2
        if y + r > 1: pen += (y + r - 1) ** 2
    # Pairwise overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            gap = circles[i, 2] + circles[j, 2] - dist
            if gap > 0:
                pen += gap ** 2
    return lam * pen


def _force_relax(circles, steps=200, lr=0.01):
    """Simple force-directed relaxation to separate overlapping circles."""
    circles = circles.copy()
    n = len(circles)
    for step in range(steps):
        forces = np.zeros_like(circles)
        # Wall repulsion
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0: forces[i, 0] += 1.0
            if y - r < 0: forces[i, 1] += 1.0
            if x + r > 1: forces[i, 0] -= 1.0
            if y + r > 1: forces[i, 1] -= 1.0
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    dx, dy = 0.01, 0.01
                    dist = np.sqrt(dx * dx + dy * dy)
                overlap = circles[i, 2] + circles[j, 2] - dist
                if overlap > 0:
                    fx = dx / dist * overlap
                    fy = dy / dist * overlap
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        # Update
        for i in range(n):
            circles[i, 0] += lr * forces[i, 0]
            circles[i, 1] += lr * forces[i, 1]
            # Clamp to [0,1]
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
    return circles


def _local_search(circles, max_iter=500, tol=1e-8):
    """Constraint-aware local search: maximize radii while maintaining validity."""
    circles = circles.copy()
    n = len(circles)
    best_sum = _sum_radii(circles)
    
    for it in range(max_iter):
        improved = False
        # Try increasing radius of each circle slightly
        for i in range(n):
            dr = 1e-4
            trial = circles.copy()
            trial[i, 2] += dr
            if _is_valid(trial):
                circles = trial
                best_sum = _sum_radii(circles)
                improved = True
            else:
                # Try moving center to make room for larger radius
                for dx in [-0.02, 0.02]:
                    for dy in [-0.02, 0.02]:
                        trial = circles.copy()
                        trial[i, 0] += dx
                        trial[i, 1] += dy
                        trial[i, 2] += dr
                        if _is_valid(trial):
                            circles = trial
                            best_sum = _sum_radii(circles)
                            improved = True
        # Try moving circles to create more space
        for i in range(n):
            for dx in [-0.01, 0.01]:
                for dy in [-0.01, 0.01]:
                    trial = circles.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    if _is_valid(trial):
                        # Check if this allows any radius increase
                        for j in range(n):
                            trial2 = trial.copy()
                            trial2[j, 2] += 1e-4
                            if _is_valid(trial2):
                                circles = trial2
                                best_sum = _sum_radii(circles)
                                improved = True
                                break
        if not improved:
            break
    return circles


def _scipy_optimize(circles, time_limit=20):
    """Use scipy to optimize positions and radii."""
    start_time = time.time()
    n = len(circles)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e10
        c = params.reshape(n, 3)
        # Radii must be positive
        c[:, 2] = np.maximum(c[:, 2], 1e-6)
        # Positions clamped
        c[:, 0] = np.clip(c[:, 0], c[:, 2], 1 - c[:, 2])
        c[:, 1] = np.clip(c[:, 1], c[:, 2], 1 - c[:, 2])
        val = _penalty(c, lam=1e3)
        return val - np.sum(c[:, 2])
    
    x0 = circles.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
    if result.success:
        c = result.x.reshape(n, 3)
        c[:, 2] = np.maximum(c[:, 2], 1e-6)
        c[:, 0] = np.clip(c[:, 0], c[:, 2], 1 - c[:, 2])
        c[:, 1] = np.clip(c[:, 1], c[:, 2], 1 - c[:, 2])
        if _is_valid(c):
            return c
    return circles


def _grid_init(n):
    """Create a grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    return np.array(cells[:n])


def _random_init(n, seed=0):
    """Random initialization with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    circles = []
    for _ in range(n):
        for attempt in range(1000):
            r = rng.uniform(0.02, 0.15)
            x = rng.uniform(r, 1 - r)
            y = rng.uniform(r, 1 - r)
            # Check against existing
            ok = True
            for cx, cy, cr in circles:
                if np.sqrt((x - cx)**2 + (y - cy)**2) < r + cr + 1e-4:
                    ok = False
                    break
            if ok:
                circles.append([x, y, r])
                break
    while len(circles) < n:
        r = 0.02
        x = rng.uniform(r, 1 - r)
        y = rng.uniform(r, 1 - r)
        circles.append([x, y, r])
    return np.array(circles)


def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_grid_init(N))
    for s in range(1, 6):
        inits.append(_random_init(N, seed=s))
    
    best = None
    best_val = -1
    
    for init in inits:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Phase 1: Force relaxation
        relaxed = _force_relax(init, steps=100, lr=0.005)
        
        # Phase 2: Local search
        searched = _local_search(relaxed, max_iter=200)
        
        # Phase 3: Scipy optimization
        if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
            optimized = _scipy_optimize(searched, time_limit=10)
        else:
            optimized = searched
        
        # Phase 4: Final polish
        polished = _local_search(optimized, max_iter=100)
        
        val = _sum_radii(polished)
        if val > best_val:
            best_val = val
            best = polished
    
    if best is None:
        # Fallback to grid
        return _grid_init(N)
    
    return best
# EVOLVE-BLOCK-END
