import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute all triangle index combinations
_tri_indices = np.array(list(combinations(range(N), 3)))

def _all_areas_cart(pts):
    """Vectorized computation of all 165 triangle areas from cartesian points."""
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area_cart(pts):
    """Exact minimum area from cartesian points."""
    return float(np.min(_all_areas_cart(pts)))

def _is_inside_triangle(pts, tol=1e-9):
    """Check if all points are inside the unit equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.all(u >= -tol) and np.all(v >= -tol) and np.all(w >= -tol)

def _objective_cart(flat_pts):
    """
    Objective function for optimization.
    Returns negative of minimum area.
    Penalizes points outside the triangle heavily.
    """
    pts = flat_pts.reshape(-1, 2)
    if not _is_inside_triangle(pts):
        return -100.0
    return -_min_area_cart(pts)

def _soft_objective_cart(flat_pts, beta=50.0):
    """
    Smoothed objective for gradient-based methods.
    Uses log-sum-exp to approximate the minimum.
    """
    pts = flat_pts.reshape(-1, 2)
    if not _is_inside_triangle(pts):
        return -100.0
    areas = _all_areas_cart(pts)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return -(m - log_sum / beta)

def _init_hexagonal_cart():
    """
    Generate a hexagonal grid initialization in Cartesian coordinates.
    """
    # Hexagonal lattice points in the equilateral triangle
    # This is a heuristic start, but DE will explore globally.
    pts = []
    # Row 0
    pts.append([0.0, 0.0])
    pts.append([0.5, 0.0])
    pts.append([1.0, 0.0])
    # Row 1
    pts.append([0.25, H/4])
    pts.append([0.75, H/4])
    # Row 2
    pts.append([0.5, H/2])
    # Row 3
    pts.append([0.25, 3*H/4])
    pts.append([0.75, 3*H/4])
    # Row 4
    pts.append([0.5, H])
    # We have 9 points. Need 11.
    # Add two more in the middle or adjust grid.
    # Let's just use random for the other 2 or split existing.
    # Actually, let's use a denser grid or just random.
    # For DE, the initial population is random anyway, so this is just a "seed" if we use it.
    # But DE generates its own population.
    return np.array(pts[:11]) # If less than 11, pad with random? 
    # Better to just let DE handle it. This function is not strictly necessary for DE 
    # unless we want to bias the initial population, which DE doesn't easily support.
    # We'll just use DE's random initialization.

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin

    def time_left():
        return deadline - time.time()

    # Define bounds for DE
    # x in [0, 1], y in [0, H]
    bounds = [(0, 1), (0, H)] * N

    # Phase 1: Differential Evolution
    # DE is stochastic and global.
    # maxiter controls the number of generations.
    # We set a callback to stop if time is running out.
    
    def de_callback(xk, convergence):
        if time_left() < 20.0: # Leave 20s for polish and overhead
            return True
        return False

    try:
        # popsize=20, tol=1e-8, maxiter=1000
        # We don't know how many iterations will fit in time, so we rely on callback.
        # However, DE's callback doesn't guarantee immediate stop in all scipy versions? 
        # It returns True to stop.
        
        # To be safe, we can wrap DE in a loop or just let it run and check time.
        # SciPy DE callback stops the iteration loop.
        
        res_de = differential_evolution(
            _objective_cart,
            bounds,
            maxiter=1000,
            popsize=25,
            tol=1e-10,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=False, # We will polish manually
            callback=de_callback,
            update_interval=10 # Check callback every 10 iterations
        )
        
        best_pts_de = res_de.x.reshape(-1, 2)
        best_score_de = -res_de.fun
        
    except Exception:
        # Fallback if DE fails
        best_pts_de = np.random.uniform(0, 1, (N, 2))
        best_pts_de[:, 1] *= H * (1 - best_pts_de[:, 0]) # Rough mapping
        # Ensure inside
        # Just use a simple random inside
        pts = []
        for _ in range(N):
            r1 = np.random.random()
            r2 = np.random.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            x = r1 + 0.5 * r2
            y = r2 * H
            pts.append([x, y])
        best_pts_de = np.array(pts)
        best_score_de = _min_area_cart(best_pts_de)

    # Phase 2: Local Polish
    # Use Nelder-Mead on the hard objective
    # Or L-BFGS-B on the soft objective
    
    current_pts = best_pts_de
    current_score = best_score_de
    
    if time_left() > 10.0:
        try:
            # Nelder-Mead is good for non-smooth functions
            res_nm = minimize(
                _objective_cart,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_nm.fun < -10: # Valid point
                pts_nm = res_nm.x.reshape(-1, 2)
                score_nm = -res_nm.fun
                if score_nm > current_score:
                    current_pts = pts_nm
                    current_score = score_nm
        except Exception:
            pass

    # Phase 3: High-precision polish with L-BFGS-B on soft objective
    if time_left() > 5.0:
        try:
            # Use soft objective for smooth gradients
            res_lf = minimize(
                _soft_objective_cart,
                current_pts.flatten(),
                args=(100.0,), # High beta for sharpness
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_lf.fun < -10:
                pts_lf = res_lf.x.reshape(-1, 2)
                # Verify it's inside and calculate hard score
                if _is_inside_triangle(pts_lf):
                    score_lf = _min_area_cart(pts_lf)
                    if score_lf > current_score:
                        current_pts = pts_lf
                        current_score = score_lf
        except Exception:
            pass

    # Final check
    if not _is_inside_triangle(current_pts):
        # If somehow outside, project or fallback
        # Simple fallback: random
        pts = []
        for _ in range(N):
            r1 = np.random.random()
            r2 = np.random.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            x = r1 + 0.5 * r2
            y = r2 * H
            pts.append([x, y])
        current_pts = np.array(pts)

    return current_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_min_area_cart(pts)}")
