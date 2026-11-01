import numpy as np
import time
from scipy.optimize import minimize

N = 32
MASK = np.triu(np.ones((N, N), dtype=bool), k=1)
I_IDX, J_IDX = np.nonzero(MASK)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def unpack(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return x, y, r

    def is_valid(v, tol=1e-9):
        x, y, r = unpack(v)
        if np.any(r < -tol):
            return False
        if np.any(x < r - tol) or np.any(1 - x < r - tol):
            return False
        if np.any(y < r - tol) or np.any(1 - y < r - tol):
            return False
        dx = x[I_IDX] - x[J_IDX]
        dy = y[I_IDX] - y[J_IDX]
        d = np.sqrt(dx*dx + dy*dy)
        if np.any(d < r[I_IDX] + r[J_IDX] - tol):
            return False
        return True

    # Soft repulsive potential: penalizes overlaps via smooth 1/d^4 term
    # plus quadratic penalty for boundary violations
    def potential(v, beta=20.0, lam_b=500.0):
        x, y, r = unpack(v)
        dx = x[I_IDX] - x[J_IDX]
        dy = y[I_IDX] - y[J_IDX]
        d2 = dx*dx + dy*dy
        d = np.sqrt(d2 + 1e-12)
        # Repulsive force: strong when d < r_i + r_j
        # Use 1/d^4 style: potential = (r_i+r_j)^4 / d^4 when d < r_i+r_j, else 0
        # Smooth version: max(0, (r_i+r_j) - d) -> use softplus-like
        # Easier: penalize (r_i+r_j - d)^2 when positive, plus 1/d^6 for very close
        r_sum = r[I_IDX] + r[J_IDX]
        overlap = np.maximum(0.0, r_sum - d)
        E_overlap = np.sum(overlap * overlap)
        # Additional strong repulsion for near-contact
        E_repel = np.sum(r_sum**4 / (d2 + 1e-8)**2)
        
        # Boundary
        pen_x = np.sum(np.maximum(0, r - x)**2)
        pen_1mx = np.sum(np.maximum(0, r - (1 - x))**2)
        pen_y = np.sum(np.maximum(0, r - y)**2)
        pen_1my = np.sum(np.maximum(0, r - (1 - y))**2)
        E_bound = pen_x + pen_1mx + pen_y + pen_1my
        
        return -np.sum(r) + beta * E_repel + lam_b * E_bound + 100.0 * E_overlap

    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N

    def get_guesses():
        guesses = []
        # Grid
        side = int(np.ceil(np.sqrt(N)))
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.flatten()[:N]
        gy = gy.flatten()[:N]
        gr = np.full(N, 0.5 / side * 0.95)
        guesses.append(np.concatenate([gx, gy, gr]))

        # Perturbed grid
        for scale in [0.01, 0.02, 0.05]:
            v = np.concatenate([gx, gy, gr]) + np.random.randn(3*N) * scale
            v = np.clip(v, [b[0] for b in bounds], [b[1] for b in bounds])
            guesses.append(v)

        # Random
        for _ in range(3):
            x = np.random.rand(N)
            y = np.random.rand(N)
            r = np.random.rand(N) * 0.08 + 0.02
            guesses.append(np.concatenate([x, y, r]))

        # Voronoi-inspired: start with grid but vary radii
        v_vor = np.concatenate([gx, gy, np.random.rand(N) * 0.04 + 0.06])
        v_vor = np.clip(v_vor, [b[0] for b in bounds], [b[1] for b in bounds])
        guesses.append(v_vor)

        return guesses

    best_score = 0.0
    best_solution = None

    guesses = get_guesses()

    for v0 in guesses:
        if time.time() > deadline - 30:
            break
        try:
            # Coarse optimization
            res = minimize(potential, v0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
            if time.time() > deadline - 10:
                break
            v_opt = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
            if is_valid(v_opt):
                score = np.sum(v_opt[2*N:3*N])
                if score > best_score:
                    best_score = score
                    best_solution = v_opt.copy()
        except Exception:
            pass

    # Final polish: high-precision local solve on best solution
    if best_solution is not None and time.time() < deadline - 5:
        try:
            def polish_obj(v):
                x, y, r = unpack(v)
                dx = x[I_IDX] - x[J_IDX]
                dy = y[I_IDX] - y[J_IDX]
                d2 = dx*dx + dy*dy
                d = np.sqrt(d2 + 1e-12)
                r_sum = r[I_IDX] + r[J_IDX]
                overlap = np.maximum(0.0, r_sum - d)
                E_overlap = np.sum(overlap * overlap)
                pen_x = np.sum(np.maximum(0, r - x)**2)
                pen_1mx = np.sum(np.maximum(0, r - (1 - x))**2)
                pen_y = np.sum(np.maximum(0, r - y)**2)
                pen_1my = np.sum(np.maximum(0, r - (1 - y))**2)
                E_bound = pen_x + pen_1mx + pen_y + pen_1my
                return -np.sum(r) + 10000.0 * (E_overlap + E_bound)

            res2 = minimize(polish_obj, best_solution, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 10000})
            v_pol = np.clip(res2.x, [b[0] for b in bounds], [b[1] for b in bounds])
            if is_valid(v_pol):
                score2 = np.sum(v_pol[2*N:3*N])
                if score2 > best_score:
                    best_score = score2
                    best_solution = v_pol
        except Exception:
            pass

    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r = unpack(best_solution)
    return np.column_stack((x, y, r))
