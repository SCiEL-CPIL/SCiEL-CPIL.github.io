"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    best_value = -np.inf
    best_result = None

    # We'll optimize: positions of N circles + aspect ratio w (h = 2-w)
    # All circles have equal radius r (to maximize sum = N*r)
    # Variables: [x_1, y_1, ..., x_N, y_N, r, w]

    def solve_for_aspect(w_init, time_budget=30):
        h_init = 2.0 - w_init
        side = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / side))

        # Estimate initial radius
        r_est = min(w_init / (2*side), h_init / (2*rows)) * 0.85

        # Build initial guess
        v0 = []
        for i in range(N):
            row = i // side
            col = i % side
            if side > 1:
                x = r_est + col * (w_init - 2*r_est) / (side - 1)
            else:
                x = w_init / 2
            if rows > 1:
                y = r_est + row * (h_init - 2*r_est) / (rows - 1)
            else:
                y = h_init / 2
            v0.extend([x, y])
        v0.extend([r_est, w_init])

        def objective(v):
            return -v[2*N]  # minimize -r

        constraints = []
        # Boundary constraints
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*2] - v[2*N]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+1] - v[i*2] - v[2*N]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*2+1] - v[2*N]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 2.0 - v[2*N+1] - v[i*2+1] - v[2*N]})
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: 
                    (v[i*2]-v[j*2])**2 + (v[i*2+1]-v[j*2+1])**2 - 4*v[2*N]**2})
        # Aspect ratio bounds
        constraints.append({'type': 'ineq', 'fun': lambda v: v[2*N+1] - 0.4})
        constraints.append({'type': 'ineq', 'fun': lambda v: 1.6 - v[2*N+1]})

        bounds = []
        for i in range(N):
            bounds.append((0.001, 3.99))
            bounds.append((0.001, 3.99))
        bounds.append((0.001, 1.0))
        bounds.append((0.4, 1.6))

        start = time.time()
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                          constraints=constraints, 
                          options={'maxiter': 300, 'ftol': 1e-9, 'disp': False})
        except Exception:
            return None, None

        elapsed = time.time() - start
        if time.time() > deadline - 5:
            return None, None

        r_opt = res.x[2*N]
        w_opt = res.x[2*N+1]
        value = N * r_opt

        # Verify feasibility
        feasible = True
        for i in range(N):
            xi, yi = res.x[i*2], res.x[i*2+1]
            if xi < r_opt - 1e-6 or w_opt - xi < r_opt - 1e-6:
                feasible = False
                break
            if yi < r_opt - 1e-6 or (2-w_opt) - yi < r_opt - 1e-6:
                feasible = False
                break
        if feasible:
            for i in range(N):
                for j in range(i+1, N):
                    dist2 = (res.x[i*2]-res.x[j*2])**2 + (res.x[i*2+1]-res.x[j*2+1])**2
                    if dist2 < 4*r_opt**2 - 1e-6:
                        feasible = False
                        break
                if not feasible:
                    break

        if not feasible:
            # Shrink r slightly to ensure feasibility
            r_opt *= 0.99
            value = N * r_opt
            # Re-verify
            for i in range(N):
                xi, yi = res.x[i*2], res.x[i*2+1]
                if xi < r_opt or w_opt - xi < r_opt or yi < r_opt or (2-w_opt) - yi < r_opt:
                    return None, None
            for i in range(N):
                for j in range(i+1, N):
                    dist2 = (res.x[i*2]-res.x[j*2])**2 + (res.x[i*2+1]-res.x[j*2+1])**2
                    if dist2 < 4*r_opt**2:
                        return None, None

        result = np.array([[res.x[i*2], res.x[i*2+1], r_opt] for i in range(N)])
        return value, result

    # Try multiple initial aspect ratios
    for w_init in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        if time.time() > deadline - 15:
            break
        value, result = solve_for_aspect(w_init)
        if result is not None and value > best_value:
            best_value = value
            best_result = result

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
