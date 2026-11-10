# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_TOL = 1e-7
_DEADLINE = 630.0


def _objective(x, n=10):
    """
    x = [s_0..s_{n-1}, cx_0..cx_{n-1}, cy_0..cy_{n-1}]
    Minimize negative sum of sides + penalties for overlap and boundary violations.
    """
    s = x[:n]
    cx = x[n:2*n]
    cy = x[2*n:3*n]
    
    # Primary objective: maximize sum of sides (minimize negative)
    obj = -np.sum(s)
    
    # Boundary penalties: squares must be in [0,1]^2
    # cx - s/2 >= 0  => s/2 - cx <= 0
    # cx + s/2 <= 1  => cx + s/2 - 1 <= 0
    # Same for y
    pen = 0.0
    lam = 1e4  # penalty weight
    
    for i in range(n):
        # left
        v = s[i]/2 - cx[i]
        if v > 0:
            pen += lam * v * v
        # right
        v = cx[i] + s[i]/2 - 1
        if v > 0:
            pen += lam * v * v
        # bottom
        v = s[i]/2 - cy[i]
        if v > 0:
            pen += lam * v * v
        # top
        v = cy[i] + s[i]/2 - 1
        if v > 0:
            pen += lam * v * v
    
    # Overlap penalties: for each pair (i,j), we need x-sep OR y-sep
    # x-sep: |cx_i - cx_j| >= (s_i + s_j)/2
    # y-sep: |cy_i - cy_j| >= (s_i + s_j)/2
    # Penalty if both violations exist:
    for i in range(n):
        for j in range(i+1, n):
            t = (s[i] + s[j]) / 2
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            # violation in x: t - dx > 0
            vx = max(0, t - dx)
            # violation in y: t - dy > 0
            vy = max(0, t - dy)
            # If both > 0, there is overlap. Penalize the product or sum.
            # Using a smooth approximation: penalty if both are positive
            # We can penalize vx * vy to encourage at least one to be zero
            # But vx*vy is 0 if either is 0. However, gradient might be tricky at 0.
            # Let's use a sum of squares of the minimum violation?
            # Actually, standard approach: penalty = (vx + vy)^2 if we want to push them apart?
            # No, we want to ensure vx==0 OR vy==0.
            # A common trick: minimize max(0, vx) + max(0, vy) is not smooth.
            # Let's use: penalty = lam_overlap * (vx * vy)
            # This is 0 if either separation is satisfied. It is positive if both are violated.
            # This encourages the optimizer to resolve at least one axis.
            pen += 1e3 * (vx * vy)
    
    return obj + pen


def _check_validity(centers, sides, n=10):
    """Check if the solution is valid according to the task contract."""
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if not np.all(sides > 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -_TOL or cx + s/2 > 1 + _TOL:
            return False
        if cy - s/2 < -_TOL or cy + s/2 > 1 + _TOL:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - _TOL
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                return False
    return True


def squares_in_square10() -> tuple:
    n = 10
    t0 = time.time()
    deadline = t0 + _DEADLINE - 30.0  # 30s safety margin
    
    best_value = 0.0
    best_centers = None
    best_sides = None
    
    # Baseline: 3x3 grid + 1 extra? Or just 3x3 grid with 10 squares?
    # 3x3 grid has 9 cells. 10th square needs to go somewhere.
    # Let's start with a simple valid configuration: 10 squares in a 5x2 grid
    # Side = 1/5 = 0.2. Sum = 2.0.
    # Or 3x3 grid (9 squares) side=1/3, sum=3.0. Plus 1 tiny square?
    # Let's use the 3x3 grid as initial best.
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    # For the 10th square (index 9), it's in row 3, col 0. But we only have 3 rows (0,1,2).
    # So row 3 is out of bounds.
    # Let's construct a valid baseline manually.
    # 3x3 grid of 9 squares, side 1/3.
    # The 10th square can be very small, placed in a corner?
    # Or better: 5x2 grid. Side 0.2. Sum 2.0.
    # Or: 3 rows. Row 0: 3 squares. Row 1: 3 squares. Row 2: 3 squares.
    # That's 9.
    # Let's just use a 5x2 grid for a safe valid baseline.
    cols = 5
    rows = 2
    side = 1.0 / cols
    cx = (np.arange(n) % cols + 0.5) * side
    cy = (np.arange(n) // cols + 0.5) * (1.0 / rows)
    best_centers = np.column_stack([cx, cy])
    best_sides = np.full(n, side)
    best_value = np.sum(best_sides)
    
    # Random restarts
    np.random.seed(42)
    num_restarts = 200  # Max number of restarts, will break on time
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial guess
        # Sides: random between 0.1 and 0.5
        s0 = np.random.uniform(0.1, 0.5, n)
        cx0 = np.random.uniform(0.1, 0.9, n)
        cy0 = np.random.uniform(0.1, 0.9, n)
        x0 = np.concatenate([s0, cx0, cy0])
        
        # Bounds: s in (0, 1], cx in [0,1], cy in [0,1]
        bounds = []
        for i in range(n):
            bounds.append((1e-6, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        
        try:
            res = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
        except Exception:
            continue
        
        if res.success or res.status == 0:
            x = res.x
            s = x[:n]
            cx = x[n:2*n]
            cy = x[2*n:3*n]
            centers = np.column_stack([cx, cy])
            
            # Check validity
            if _check_validity(centers, s):
                val = np.sum(s)
                if val > best_value:
                    best_value = val
                    best_centers = centers
                    best_sides = s
    
    # Final check
    if best_centers is None:
        # Fallback to 5x2 grid
        cols = 5
        rows = 2
        side = 1.0 / cols
        cx = (np.arange(n) % cols + 0.5) * side
        cy = (np.arange(n) // cols + 0.5) * (1.0 / rows)
        best_centers = np.column_stack([cx, cy])
        best_sides = np.full(n, side)
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
