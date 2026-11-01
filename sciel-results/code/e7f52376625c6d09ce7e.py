from scipy.optimize import minimize
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar, differential_evolution

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _generate_rigid_body(s, theta):
    """
    Generate 14 points: 6 vertices of a regular octahedron (radius 1)
    and 8 vertices of a regular cube (radius s) rotated by theta around z-axis.
    """
    # Octahedron vertices: (±1,0,0), (0,±1,0), (0,0,±1)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    # Cube vertices: (±1,±1,±1) scaled by s/sqrt(3) to have circumradius s?
    # The prompt says "scaling factor". Let's define s as the circumradius of the cube.
    # Standard cube vertices are (±1, ±1, ±1). Norm is sqrt(3).
    # To make circumradius s, we scale by s/sqrt(3).
    # However, it's often easier to think of s as the half-side length or just a free parameter.
    # Let's use s as the multiplier for the standard (±1,±1,±1) coordinates.
    # Then the distance from origin to cube vertex is s*sqrt(3).
    # Distance between adjacent cube vertices is 2*s.
    # Distance between opposite cube vertices is 2*s*sqrt(3).
    
    cube_base = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    
    # Scale cube by s
    cube = cube_base * s
    
    # Rotate cube around z-axis by theta
    c = np.cos(theta)
    s_ = np.sin(theta)
    R = np.array([
        [c, -s_, 0],
        [s_, c, 0],
        [0, 0, 1]
    ])
    
    cube_rot = (R @ cube.T).T
    
    pts = np.vstack([octa, cube_rot])
    return pts

def _objective(params):
    s, theta = params
    pts = _generate_rigid_body(s, theta)
    return _true_score(pts)

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    # The ratio is scale invariant, so the absolute scale of the whole set doesn't matter.
    # However, the relative scale between the octahedron (fixed radius 1) and the cube (scale s) matters.
    # The octahedron has min distance 1 (between adjacent vertices? No, (1,0,0) to (0,1,0) is sqrt(2)).
    # Octahedron min dist: sqrt(2) (e.g. (1,0,0) to (0,1,0)).
    # Octahedron max dist: 2 (e.g. (1,0,0) to (-1,0,0)).
    
    # We need to find optimal s and theta.
    # s should be in a reasonable range. 
    # If s is too small, cube points are close to origin, potentially creating small distances to octa points.
    # If s is too large, cube points are far, creating large distances.
    
    # Let's define the search space.
    # s: 0.1 to 2.0
    # theta: 0 to pi/2 (symmetry of cube)
    
    bounds = [(0.1, 2.0), (0.0, np.pi/2)]
    
    best_score = np.inf
    best_params = None
    
    # 1. Differential Evolution for global search
    # DE is robust and handles non-smooth functions well.
    # Since the function is periodic and symmetric, we restrict theta to [0, pi/2].
    # We also need to consider that the "rigid body" assumption might be slightly broken in the true optimum,
    # but this is a good starting point.
    
    try:
        # Use DE with a reasonable population size and max generations
        # Keep it fast: popsize=10, maxiter=50
        res_de = differential_evolution(
            _objective, 
            bounds, 
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-6,
            polish=False # We'll do polish manually to control time
        )
        best_params = res_de.x
        best_score = res_de.fun
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to grid search if DE fails
        s_vals = np.linspace(0.2, 1.8, 10)
        theta_vals = np.linspace(0, np.pi/2, 10)
        for s in s_vals:
            for theta in theta_vals:
                sc = _objective([s, theta])
                if sc < best_score:
                    best_score = sc
                    best_params = [s, theta]
    
    # 2. Local refinement using minimize_scalar on theta for fixed s, and vice versa?
    # Or just use Nelder-Mead on the 2D parameter space.
    
    if time.time() - _t0 > DEADLINE - 10:
        pts = _generate_rigid_body(best_params[0], best_params[1])
        return pts

    try:
        # Refine with Nelder-Mead
        res_nm = minimize(
            _objective, 
            best_params, 
            method='Nelder-Mead', 
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_params = res_nm.x
    except Exception:
        pass
    
    # 3. Check alternative symmetries?
    # The cube rotation is around Z. The octahedron is aligned with axes.
    # Due to symmetry, rotating the whole system doesn't change the score.
    # But rotating the cube relative to the octahedron is what we are doing.
    # Are there other high-symmetry configurations?
    # E.g., Cube vertices at (±1,0,0)... no that's octahedron.
    # What if we swap roles? Cube core, Octahedron shell?
    # That is effectively s -> 1/s with some rotation?
    # Let's try a second DE run with a different initial bias or just rely on the wide bounds.
    
    # To be safe, let's try a few specific "snapshots" that might be local minima
    # e.g. theta = 0, pi/4, etc.
    
    candidates = [best_params.copy()]
    
    # Add some manual candidates
    candidates.append([1.0, 0.0])
    candidates.append([1.0, np.pi/4])
    candidates.append([0.8, np.pi/4])
    candidates.append([1.2, np.pi/4])
    
    for cand in candidates:
        if time.time() - _t0 > DEADLINE - 5:
            break
        try:
            res_c = minimize(
                _objective, 
                cand, 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_c.fun < best_score:
                best_score = res_c.fun
                best_params = res_c.x
        except Exception:
            pass
            
    pts = _generate_rigid_body(best_params[0], best_params[1])
    
    # Final check: is the score valid?
    sc = _true_score(pts)
    if not np.isfinite(sc):
        # Fallback to a known good config if something went wrong
        # E.g. Fibonacci sphere
        i = np.arange(N)
        phi = np.arccos(1 - 2 * (i + 0.5) / N)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts = np.column_stack([x, y, z])
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
    return pts

if __name__ == "__main__":
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Score: {_true_score(pts):.4f}")
    print(pts)
