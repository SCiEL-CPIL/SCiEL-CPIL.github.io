"""Circle packing in unit square using Apollonian-gasket-inspired construction."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, atan2, sin, cos, pi

_DEADLINE = 140.0  # safety margin before 150s kill


def _validate(circles):
    """Check that circles are valid (in bounds, non-overlapping)."""
    for (x, y, r) in circles:
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
        if r <= 0:
            return False
    
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            d = sqrt((x1-x2)**2 + (y1-y2)**2)
            if d < r1 + r2 - 1e-9:
                return False
    return True


def _objective(circles):
    """Sum of radii."""
    return sum(r for (_, _, r) in circles)


def _grid_init(n):
    """Initialize n circles on a grid with small radii."""
    side = int(np.ceil(np.sqrt(n)))
    circles = []
    for i in range(side):
        for j in range(side):
            if len(circles) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.3 / side  # conservative initial radius
            circles.append((x, y, r))
    return circles


def _physics_optimize(circles, max_steps=50000, dt=0.001, k_repel=10.0, k_wall=50.0, seed=42):
    """
    Physics-based optimization using repulsion forces.
    Circles repel each other when overlapping and are attracted to maximize radius.
    """
    rng = np.random.default_rng(seed)
    circles = [list(c) for c in circles]  # make mutable copies
    n = len(circles)
    
    best_circles = None
    best_obj = _objective(circles)
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 3:
            break
        
        # Compute forces
        forces = np.zeros((n, 2))
        
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    # Random nudge
                    forces[i] += rng.normal(0, 0.01, 2)
                    forces[j] -= rng.normal(0, 0.01, 2)
                    continue
                
                r_sum = circles[i][2] + circles[j][2]
                if dist < r_sum:
                    # Overlap: repel proportional to overlap
                    overlap = r_sum - dist
                    force_mag = k_repel * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]
        
        # Wall constraints: push circles inward if they extend beyond boundary
        for i in range(n):
            x, y, r = circles[i]
            # Left wall
            if x - r < 0:
                forces[i][0] += k_wall * (r - x)
            # Right wall
            if x + r > 1:
                forces[i][0] -= k_wall * (x + r - 1)
            # Bottom wall
            if y - r < 0:
                forces[i][1] += k_wall * (r - y)
            # Top wall
            if y + r > 1:
                forces[i][1] -= k_wall * (y + r - 1)
        
        # Apply forces to update positions
        for i in range(n):
            circles[i][0] += dt * forces[i][0]
            circles[i][1] += dt * forces[i][1]
        
        # Add small random perturbation (simulated annealing)
        if step % 100 == 0:
            temp = 0.01 * (1 - step / max_steps)
            for i in range(n):
                circles[i][0] += rng.normal(0, temp)
                circles[i][1] += rng.normal(0, temp)
        
        # Check if valid and update best
        if _validate(circles):
            obj = _objective(circles)
            if obj > best_obj:
                best_obj = obj
                best_circles = [tuple(c) for c in circles]
    
    return best_circles if best_circles is not None else circles


def _construct_physics():
    """Construct using physics-based optimization."""
    # Start with grid initialization
    circles = _grid_init(N)
    
    # Run physics optimization
    circles = _physics_optimize(circles, max_steps=30000, dt=0.002, k_repel=15.0, k_wall=60.0)
    
    # Local refinement: try to increase radii where possible
    for i in range(N):
        if time.time() > _DEADLINE - 2:
            break
        x, y, r = circles[i]
        # Find max possible radius at this position
        r_max = min(x, 1-x, y, 1-y)
        for j in range(N):
            if i == j:
                continue
            dx = x - circles[j][0]
            dy = y - circles[j][1]
            d = sqrt(dx*dx + dy*dy)
            r_max = min(r_max, d - circles[j][2])
        r_max = max(0, r_max)
        if r_max > r * 1.001:
            circles[i] = (x, y, r_max * 0.999)
    
    return circles


def _construct_grid_fallback():
    """Fallback grid construction."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return cells[:N]


def construct():
    start_time = time.time()
    
    try:
        circles = _construct_physics()
        
        if len(circles) != N:
            circles = _construct_grid_fallback()
        
        # Validate
        if not _validate(circles):
            # Try to fix by reducing radii slightly
            circles_fixed = [(x, y, max(1e-5, r * 0.95)) for (x, y, r) in circles]
            if not _validate(circles_fixed):
                circles = _construct_grid_fallback()
            else:
                circles = circles_fixed
    except Exception:
        circles = _construct_grid_fallback()
    
    # Ensure exactly N circles
    if len(circles) > N:
        circles = circles[:N]
    elif len(circles) < N:
        while len(circles) < N:
            circles.append((0.5, 0.5, 1e-5))
    
    result = np.array(circles, dtype=np.float64)
    
    # Final validation
    if not _validate(result):
        result = np.array(_construct_grid_fallback(), dtype=np.float64)
    
    return result
# EVOLVE-BLOCK-END
