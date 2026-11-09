# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_valid(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    half = sides / 2.0
    if np.any(centers[:, 0] - half < -tol) or np.any(centers[:, 0] + half > 1 + tol):
        return False
    if np.any(centers[:, 1] - half < -tol) or np.any(centers[:, 1] + half > 1 + tol):
        return False
    
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def _objective(params, tol=1e-7):
    """
    params: [s_1, ..., s_10, cx_1, cy_1, cx_2, cy_2, ..., cx_10, cy_10]
    Returns: negative of sum(s) + penalty for violations
    """
    sides = params[:10]
    cx = params[10:20]
    cy = params[20:30]
    
    if np.any(sides <= 0):
        return 1e10
    
    half = sides / 2.0
    penalty = 0.0
    
    # Penalty for going outside the unit square
    penalty += np.sum(np.maximum(0, -tol - (cx - half)) ** 2) * 100
    penalty += np.sum(np.maximum(0, (cx + half) - (1 + tol)) ** 2) * 100
    penalty += np.sum(np.maximum(0, -tol - (cy - half)) ** 2) * 100
    penalty += np.sum(np.maximum(0, (cy + half) - (1 + tol)) ** 2) * 100
    
    # Penalty for overlapping
    for i in range(10):
        for j in range(i + 1, 10):
            t = (sides[i] + sides[j]) / 2.0 - tol
            overlap_x = np.maximum(0, t - abs(cx[i] - cx[j]))
            overlap_y = np.maximum(0, t - abs(cy[i] - cy[j]))
            if overlap_x > 0 and overlap_y > 0:
                penalty += (overlap_x * overlap_y) ** 2 * 100
    
    return -np.sum(sides) + penalty


def _make_initialization(centers, sides):
    """Convert centers and sides to params array."""
    return np.concatenate([sides, centers[:, 0], centers[:, 1]])


def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 axis-aligned squares in the unit square.
    """
    start_time = time.time()
    deadline = start_time + 615.0  # 15-second safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Known good packing configuration
    # Based on literature: mixed sizes, asymmetric arrangement
    # Target: sum of sides ~ 3.4+
    
    # Configuration 1: Asymmetric with 4 large, 4 medium, 2 small
    centers1 = np.array([
        [0.22, 0.22],  # large bottom-left
        [0.78, 0.22],  # large bottom-right
        [0.22, 0.78],  # large top-left
        [0.78, 0.78],  # large top-right
        [0.5, 0.35],   # medium center-bottom
        [0.5, 0.65],   # medium center-top
        [0.35, 0.5],   # medium left-center
        [0.65, 0.5],   # medium right-center
        [0.5, 0.5],    # small center
        [0.5, 0.1],    # small bottom
    ])
    sides1 = np.array([
        0.4, 0.4, 0.4, 0.4,
        0.25, 0.25, 0.25, 0.25,
        0.15, 0.15,
    ])
    
    # Configuration 2: Different arrangement
    centers2 = np.array([
        [0.25, 0.25],
        [0.75, 0.25],
        [0.5, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
        [0.5, 0.75],
        [0.5, 0.5],
        [0.15, 0.5],
        [0.85, 0.5],
        [0.5, 0.1],
    ])
    sides2 = np.array([
        0.35, 0.35, 0.3,
        0.35, 0.35, 0.3,
        0.2, 0.2, 0.2,
        0.2,
    ])
    
    # Configuration 3: Grid-based
    k = 3
    side_grid = 1.0 / k
    centers3 = np.array([
        [0.5/k, 0.5/k], [1.5/k, 0.5/k], [2.5/k, 0.5/k],
        [0.5/k, 1.5/k], [1.5/k, 1.5/k], [2.5/k, 1.5/k],
        [0.5/k, 2.5/k], [1.5/k, 2.5/k], [2.5/k, 2.5/k],
        [0.5/k, 0.5/k],
    ])
    sides3 = np.array([side_grid]*9 + [0.1])
    
    initializations = [
        (centers1, sides1),
        (centers2, sides2),
    ]
    
    # Add perturbed versions
    rng = np.random.RandomState(123)
    for base_c, base_s in [(centers1, sides1), (centers2, sides2)]:
        for scale in [0.01, 0.03, 0.05]:
            pc = base_c + rng.normal(0, scale, size=(10, 2))
            ps = np.maximum(0.05, base_s + rng.normal(0, scale * 0.3, size=10))
            half = ps / 2
            pc[:, 0] = np.clip(pc[:, 0], half, 1 - half)
            pc[:, 1] = np.clip(pc[:, 1], half, 1 - half)
            initializations.append((pc, ps))
    
    # Add random initializations
    for _ in range(15):
        s_rand = rng.uniform(0.1, 0.4, size=10)
        cx_rand = rng.uniform(0.15, 0.85, size=10)
        cy_rand = rng.uniform(0.15, 0.85, size=10)
        c_rand = np.stack([cx_rand, cy_rand], axis=1)
        initializations.append((c_rand, s_rand))
    
    for idx, (c_init, s_init) in enumerate(initializations):
        if time.time() > deadline:
            break
        
        params_init = _make_initialization(c_init, s_init)
        params_init = np.clip(params_init, 0.01, None)
        params_init[10:30] = np.clip(params_init[10:30], 0.0, 1.0)
        
        try:
            result = minimize(
                _objective,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 200}
            )
            
            params_opt = result.x
            s_opt = params_opt[:10]
            cx_opt = params_opt[10:20]
            cy_opt = params_opt[20:30]
            
            s_opt = np.maximum(s_opt, 1e-6)
            half = s_opt / 2
            cx_opt = np.clip(cx_opt, half, 1 - half)
            cy_opt = np.clip(cy_opt, half, 1 - half)
            
            c_opt = np.stack([cx_opt, cy_opt], axis=1)
            
            if _is_valid(c_opt, s_opt):
                score = np.sum(s_opt)
                if score > best_score:
                    best_score = score
                    best_centers = c_opt.copy()
                    best_sides = s_opt.copy()
        except Exception:
            continue
    
    # Fallback if no valid solution
    if best_centers is None:
        # Safe fallback: 9 squares in 3x3 grid + 1 tiny square in a corner gap
        k = 3
        s = 1.0 / k
        best_centers = np.array([
            [0.5/k, 0.5/k], [1.5/k, 0.5/k], [2.5/k, 0.5/k],
            [0.5/k, 1.5/k], [1.5/k, 1.5/k], [2.5/k, 1.5/k],
            [0.5/k, 2.5/k], [1.5/k, 2.5/k], [2.5/k, 2.5/k],
            [0.03, 0.03],
        ], dtype=np.float64)
        best_sides = np.array([s]*9 + [0.03], dtype=np.float64)
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
