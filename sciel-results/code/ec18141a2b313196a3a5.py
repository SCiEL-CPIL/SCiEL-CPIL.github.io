import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

# Pre-compute indices for all triples
_IDX_I = []
_IDX_J = []
_IDX_K = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _IDX_I.append(i)
            _IDX_J.append(j)
            _IDX_K.append(k)
_IDX_I = np.array(_IDX_I)
_IDX_J = np.array(_IDX_J)
_IDX_K = np.array(_IDX_K)

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
            
        if hull_area < 1e-10:
            return 0.0
        
        # Compute triangle areas
        pi = points[_IDX_I]
        pj = points[_IDX_J]
        pk = points[_IDX_K]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def _angles_to_points(angles):
    """Convert sorted angles on unit circle to points."""
    angles = np.sort(angles) % (2 * np.pi)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _circular_objective(angles):
    """
    Objective for circular parameterization: minimize -score
    """
    try:
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            return 1e6
        points = _angles_to_points(angles)
        s = _score(points)
        if s <= 1e-15:
            return 1e6
        return -s
    except Exception:
        return 1e6

def _parab_coeffs_to_points(coeffs):
    """
    coeffs: array of shape (6,) = [a0, a1, a2, b0, b1, b2]
    t is uniformly spaced in [-1, 1] for N points.
    x = a0 + a1*t + a2*t^2
    y = b0 + b1*t + b2*t^2
    """
    t = np.linspace(-1, 1, N)
    x = coeffs[0] + coeffs[1] * t + coeffs[2] * t**2
    y = coeffs[3] + coeffs[4] * t + coeffs[5] * t**2
    return np.column_stack([x, y])

def _parab_objective(coeffs):
    """
    Objective for parabolic parameterization: minimize -score
    """
    try:
        if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
            return 1e6
        points = _parab_coeffs_to_points(coeffs)
        s = _score(points)
        if s <= 1e-15:
            return 1e6
        return -s
    except Exception:
        return 1e6

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = None
    best_score = -np.inf
    best_angles = None
    best_parab = None
    
    rng = np.random.default_rng(42)
    
    # --- Phase 1: Circular parameterization search ---
    guesses = []
    
    # 1. Regular polygon
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    guesses.append(angles_reg)
    
    # 2. Perturbed regular polygons
    for _ in range(20):
        noise = rng.normal(0, 0.5, N)
        guesses.append(angles_reg + noise)
        
    # 3. Random angles
    for _ in range(20):
        guesses.append(np.sort(rng.uniform(0, 2 * np.pi, N)))
        
    # 4. Clustered angles
    for spread in [0.5, 1.0, 2.0]:
        for _ in range(5):
            base = rng.uniform(0, 2 * np.pi, N)
            offsets = rng.uniform(-spread, spread, N)
            guesses.append(np.sort(base + offsets))

    scored_guesses = []
    for g in guesses:
        points = _angles_to_points(g)
        s = _score(points)
        scored_guesses.append((s, g))
        if s > best_score:
            best_score = s
            best_points = points.copy()
            best_angles = g.copy()
            
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top candidates with Nelder-Mead
    for idx in range(min(10, len(scored_guesses))):
        if time.time() > deadline - 15:
            break
            
        s_init, angles = scored_guesses[idx]
        
        try:
            res = minimize(
                _circular_objective,
                angles,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = _angles_to_points(res.x)
                best_angles = res.x
        except Exception:
            pass

    # Final polish on the best circular
    if best_angles is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                _circular_objective,
                best_angles,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = _angles_to_points(res.x)
                best_angles = res.x
        except Exception:
            pass

    # --- Phase 2: Parabolic parameterization search ---
    # The strategy: x = a0 + a1*t + a2*t^2, y = b0 + b1*t + b2*t^2
    # We want to explore configurations where points lie on a convex parabola-like curve.
    # Initial guess: unit circle approximated by parabola? Or just random parabolas.
    # A circle is x^2 + y^2 = 1. Locally, y ~ 1 - x^2/2. So b0=1, b2=-0.5, a1=1, others 0 is a local approx.
    # But we want global convex hull.
    # Let's try standard parabola y = x^2 (scaled) or similar.
    
    parab_guesses = []
    
    # 1. Standard parabola y = x^2, x in [-1, 1]
    # x = t, y = t^2 -> a0=0, a1=1, a2=0, b0=0, b1=0, b2=1
    parab_guesses.append(np.array([0, 1, 0, 0, 0, 1]))
    
    # 2. Inverted parabola y = 1 - x^2
    # x = t, y = 1 - t^2 -> a0=0, a1=1, a2=0, b0=1, b1=0, b2=-1
    parab_guesses.append(np.array([0, 1, 0, 1, 0, -1]))
    
    # 3. Ellipse-like approximations?
    # x = cos(t), y = sin(t) is not a parabola.
    # But we can try random parabolas.
    
    for _ in range(30):
        a0 = rng.uniform(-1, 1)
        a1 = rng.uniform(0.5, 1.5)  # ensure x spans a reasonable range
        a2 = rng.uniform(-1, 1)
        b0 = rng.uniform(-1, 1)
        b1 = rng.uniform(-1, 1)
        b2 = rng.uniform(-1, 1)
        parab_guesses.append(np.array([a0, a1, a2, b0, b1, b2]))
        
    # 4. "Fat" parabolas (wide x, narrow y) and "Tall" parabolas
    for scale_x in [0.5, 1.0, 2.0]:
        for scale_y in [0.5, 1.0, 2.0]:
            # x = scale_x * t, y = scale_y * t^2
            parab_guesses.append(np.array([0, scale_x, 0, 0, 0, scale_y]))
            # x = scale_x * t, y = scale_y * (1 - t^2)
            parab_guesses.append(np.array([0, scale_x, 0, scale_y, 0, -scale_y]))

    scored_parab_guesses = []
    for g in parab_guesses:
        points = _parab_coeffs_to_points(g)
        s = _score(points)
        scored_parab_guesses.append((s, g))
        if s > best_score:
            best_score = s
            best_points = points.copy()
            best_parab = g.copy()
            
    scored_parab_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top parabola candidates
    for idx in range(min(10, len(scored_parab_guesses))):
        if time.time() > deadline - 5:
            break
            
        s_init, coeffs = scored_parab_guesses[idx]
        
        try:
            res = minimize(
                _parab_objective,
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = _parab_coeffs_to_points(res.x)
                best_parab = res.x
        except Exception:
            pass

    # Final polish on the best parabola
    if best_parab is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                _parab_objective,
                best_parab,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = _parab_coeffs_to_points(res.x)
                best_parab = res.x
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = _angles_to_points(angles)
        
    return best_points
