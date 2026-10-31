"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time as _time

def _check_validity(circles, tol=1e-10):
    """Check if circles are valid: inside unit square and disjoint interiors."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < r1 + r2 - tol:
                return False
    return True

def _sum_radii(circles):
    return sum(c[2] for c in circles)

def construct():
    start_time = _time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    # Initial guess: grid of 26 circles
    # Use a 5x5 grid but only take 26
    side = 5
    positions = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                positions.append(((i + 0.5) / side, (j + 0.5) / side))
                radii.append(0.5 / side * 0.95)
    
    # Convert to array: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    for k in range(N):
        x0[3*k] = positions[k][0]
        x0[3*k+1] = positions[k][1]
        x0[3*k+2] = radii[k]
    
    def objective(v):
        """Negative sum of radii."""
        return -v[2::3].sum()
    
    def constraints(v):
        cons = []
        for k in range(N):
            x, y, r = v[3*k], v[3*k+1], v[3*k+2]
            # Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
            cons.append({'type': 'ineq', 'fun': lambda v, k=k: v[3*k] - v[3*k+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, k=k: 1 - v[3*k] - v[3*k+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, k=k: v[3*k+1] - v[3*k+2]})
            cons.append({'type': 'ineq', 'fun': lambda v, k=k: 1 - v[3*k+1] - v[3*k+2]})
            # Radius positive
            cons.append({'type': 'ineq', 'fun': lambda v, k=k: v[3*k+2]})
        
        # Non-overlap constraints: dist >= r1 + r2
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                # dist^2 >= (r1+r2)^2  =>  (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
                })
        return cons
    
    best_result = None
    best_value = -np.inf
    
    # Try multiple initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if _time.time() > deadline - 20:
            break
        
        # Perturb initial guess
        if restart == 0:
            v0 = x0.copy()
        else:
            v0 = x0.copy()
            # Add small random perturbations
            np.random.seed(restart * 42)
            v0[:2*N] += np.random.randn(2*N) * 0.02
            v0[2::3] *= (1 + np.random.randn(N) * 0.1)
            v0[2::3] = np.maximum(v0[2::3], 0.01)
        
        try:
            res = minimize(
                objective,
                v0,
                method='SLSQP',
                constraints=constraints,
                bounds=[(0.001, 0.5)] * 3 * N,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < best_value:
                # Check validity
                circles = np.array([res.x[3*k:3*k+3] for k in range(N)])
                if _check_validity(circles):
                    val = _sum_radii(circles)
                    if val > best_value:
                        best_value = val
                        best_result = circles
        except Exception:
            pass
    
    # If optimization didn't yield good result, fall back to initial grid
    if best_result is None:
        circles = np.array([list(p) + [r] for p, r in zip(positions, radii)])
        if not _check_validity(circles):
            # Make radii even smaller
            circles[:, 2] *= 0.9
        best_result = circles
    
    return best_result
# EVOLVE-BLOCK-END
