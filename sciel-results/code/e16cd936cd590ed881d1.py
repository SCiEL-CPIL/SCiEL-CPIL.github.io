"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 140  # seconds, safety margin from 150s limit


def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute all pairwise differences
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    # Cross product for all triples (i, j, k)
    # area(i,j,k) = 0.5 * |dx(i,j)*dy(i,k) - dy(i,j)*dx(i,k)|
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    # Extract i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA


def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle with small margin."""
    x, y = p
    eps = 1e-9
    if y < -eps or y > S3H + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    if y > S3H * (1 - x) + eps:
        return False
    if y > S3H * x + eps:
        return False
    return True


def _penalized_min_area(x):
    """Objective for minimization: minimize negative min area plus penalty for out-of-bounds."""
    points = x.reshape(-1, 2)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for i in range(N):
        px, py = points[i]
        # Distances to boundaries (positive if outside)
        d_bottom = -py
        d_top = py - S3H
        d_left = -px
        d_right = px - 1
        d_slant_right = py - S3H * (1 - px)
        d_slant_left = py - S3H * px
        
        dist = max(0.0, d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
        penalty += dist ** 2
    
    min_a = _min_area_normalized(points)
    # Large penalty to ensure points stay inside
    return -min_a + 100.0 * penalty


def construct():
    start_time = time.time()
    
    # Define bounds for Differential Evolution
    # The triangle is within [0, 1] x [0, S3H]
    bounds = [(0.0, 1.0), (0.0, S3H)] * N
    
    best_points = None
    best_score = -float('inf')
    
    # Phase 1: Global Search with Differential Evolution
    # We allocate about 100 seconds for DE, leaving 40s for polish and safety
    de_time_limit = 100
    
    try:
        # Use a callback to stop if time runs out
        def de_callback(xk, convergence):
            if time.time() - start_time > start_time + de_time_limit:
                raise RuntimeError("Time limit reached for DE")
            return False
        
        # differential_evolution minimizes the function
        # We want to maximize min_area, so we minimize -min_area
        # The function _penalized_min_area does this
        res_de = differential_evolution(
            _penalized_min_area,
            bounds,
            maxiter=200,
            popsize=20,
            tol=1e-8,
            seed=42,
            callback=de_callback,
            init='sobol' # Better initialization
        )
        
        if res_de.success or res_de.x is not None:
            pts = res_de.x.reshape(-1, 2)
            # Check validity
            valid = all(_is_inside_triangle(p) for p in pts)
            if valid:
                score = _min_area_normalized(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
            else:
                # If DE resulted in invalid points, try to project or just ignore
                # For now, if invalid, we rely on the penalty having kept it close, 
                # but we must return valid points. 
                # We can try a quick local repair or just discard.
                # Given the high penalty, it's unlikely to be far out.
                # Let's clamp and re-evaluate.
                pass
                
    except RuntimeError as e:
        if "Time limit" in str(e):
            # We need to grab the best result found so far.
            # SciPy's DE doesn't easily expose the best intermediate state in callback.
            # We can run DE in a way that we check time inside a wrapper?
            # Or just let it finish if it's close. 
            # A safer approach: run DE with a fixed small number of iterations that fits in time.
            # Or, run DE, and if it times out, we might not have the result.
            # Let's adjust strategy: Run DE with a strict iteration count that is known to be fast enough.
            # Or, use the fact that DE is deterministic with seed. 
            # Let's try to catch the exception and use a fallback or just proceed to polish if we have a candidate.
            # If DE crashes/times out without a result, we have best_points = None.
            pass
    except Exception:
        pass

    # If DE didn't produce a valid result, do a few random restarts with NM as fallback
    if best_points is None:
        rng = np.random.RandomState(42)
        def random_point_in_triangle():
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = S3H * v
                    if 0.01 < x < 0.99 and 0.01 < y < S3H * 0.99:
                        if y < S3H * (1 - x) * 0.99 and y < S3H * x * 0.99:
                            return [x, y]
        
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            x0 = np.array([random_point_in_triangle() for _ in range(N)]).flatten()
            try:
                res = minimize(_penalized_min_area, x0, method='Nelder-Mead', options={'maxiter': 500})
                pts = res.x.reshape(-1, 2)
                if all(_is_inside_triangle(p) for p in pts):
                    score = _min_area_normalized(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
            except Exception:
                continue
        
        if best_points is None:
            # Absolute fallback
            best_points = np.array([random_point_in_triangle() for _ in range(N)])
            best_score = _min_area_normalized(best_points)

    # Phase 2: Local Polish with Nelder-Mead
    if time.time() - start_time < DEADLINE - 5:
        try:
            x0 = best_points.flatten()
            res_nm = minimize(
                _penalized_min_area,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            polished = res_nm.x.reshape(-1, 2)
            if all(_is_inside_triangle(p) for p in polished):
                score = _min_area_normalized(polished)
                if score > best_score:
                    best_points = polished
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
