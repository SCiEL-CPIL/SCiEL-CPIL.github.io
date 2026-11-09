"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    def norm(vec):
        return np.linalg.norm(vec)

    def safe_F(x):
        try:
            y = F(x)
            y = np.asarray(y, dtype=float)
            if y.shape != (2,) or not np.all(np.isfinite(y)):
                return None
            return y
        except Exception:
            return None

    def is_good(x, Fx=None):
        if not np.all(np.isfinite(x)):
            return False
        if Fx is None:
            Fx = safe_F(x)
        if Fx is None:
            return False
        return norm(Fx) <= 1e-8

    # 1. Initial Evaluation
    x0 = np.asarray(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    Fx0 = safe_F(x0)
    if Fx0 is not None and norm(Fx0) <= 1e-8:
        return x0

    # 2. Local Quadratic Model Solver (Broyden + Line Search)
    # This serves as the primary solver. It uses 3 function evaluations per iteration (1 for F, 2 for Jacobian update).
    def local_solver(x_start, Fx_start, max_iter=50):
        if Fx_start is None:
            return x_start, None, 0

        x_curr = x_start.copy()
        Fx_curr = Fx_start.copy()
        calls = 0

        # Initialize Jacobian
        eps = 1e-5
        J = np.zeros((2, 2))
        valid_J = True
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            Fx_e = safe_F(x_curr + e)
            calls += 1
            if Fx_e is None:
                valid_J = False
                break
            J[:, j] = (Fx_e - Fx_curr) / eps
        
        if not valid_J:
            return x_curr, Fx_curr, calls

        best_x = x_curr.copy()
        best_Fx = Fx_curr.copy()
        best_norm = norm(Fx_curr)

        for _ in range(max_iter):
            if time_left() < 1.0:
                break

            current_norm = norm(Fx_curr)
            if current_norm < best_norm:
                best_norm = current_norm
                best_x = x_curr.copy()
                best_Fx = Fx_curr.copy()
            
            if current_norm <= 1e-10: # Tighter local tol
                return best_x, best_Fx, calls

            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx_curr)
            except np.linalg.LinAlgError:
                # Regularize if singular
                try:
                    dx = np.linalg.solve(J + 1e-4 * np.eye(2), -Fx_curr)
                except:
                    break

            if not np.all(np.isfinite(dx)):
                break

            # Line Search
            lam = 1.0
            step_accepted = False
            for _ in range(20):
                x_new = x_curr + lam * dx
                Fx_new = safe_F(x_new)
                calls += 1
                if Fx_new is None:
                    lam *= 0.5
                    continue

                new_norm = norm(Fx_new)
                if new_norm < current_norm:
                    # Broyden Update
                    delta_F = Fx_new - Fx_curr
                    J_dx = J @ dx
                    denom = np.dot(dx, dx)
                    if denom > 1e-15:
                        J += np.outer((delta_F - J_dx), dx) / denom
                    
                    x_curr = x_new
                    Fx_curr = Fx_new
                    step_accepted = True
                    break
                else:
                    lam *= 0.5
            
            if not step_accepted:
                break
            if norm(dx) < 1e-12:
                break

        return best_x, best_Fx, calls

    # Run local solver from x0
    x_best, Fx_best, _ = local_solver(x0, Fx0)
    if Fx_best is not None and norm(Fx_best) <= 1e-8:
        return x_best

    # 3. Global Search (Nelder-Mead on ||F||^2)
    # If local solver failed, the root might be far away or in a different basin.
    # We use Nelder-Mead to minimize g(x) = ||F(x)||^2.
    
    def objective(x_vec):
        Fx = safe_F(x_vec)
        if Fx is None:
            return 1e10
        return np.dot(Fx, Fx)

    # Generate diverse starting points
    scale = max(1.0, norm(x0))
    rng = np.random.default_rng(42)
    
    start_points = [x0, x_best, np.array([0.0, 0.0])]
    
    # Add points on a sphere
    for r in [scale * 0.5, scale * 1.0, scale * 2.0, scale * 5.0]:
        for _ in range(3):
            u = rng.normal(size=2)
            u /= norm(u)
            start_points.append(x0 + r * u)
            start_points.append(r * u)
            
    # Deduplicate
    unique_points = []
    for p in start_points:
        if not np.all(np.isfinite(p)):
            continue
        is_dup = False
        for up in unique_points:
            if norm(p - up) < 1e-5:
                is_dup = True
                break
        if not is_dup:
            unique_points.append(p)

    best_global_x = x_best
    best_global_Fx = Fx_best
    best_global_norm = norm(Fx_best) if Fx_best is not None else np.inf

    for sp in unique_points[:10]:
        if time_left() < 2.0:
            break
        
        # Run Nelder-Mead
        # We implement a simple NM or use scipy if available? 
        # The prompt says scipy is available.
        try:
            from scipy.optimize import minimize
            res = minimize(objective, sp, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.x is not None and np.all(np.isfinite(res.x)):
                Fx_res = safe_F(res.x)
                if Fx_res is not None:
                    n_res = norm(Fx_res)
                    if n_res < best_global_norm:
                        best_global_norm = n_res
                        best_global_x = res.x
                        best_global_Fx = Fx_res
                        if best_global_norm <= 1e-8:
                            return best_global_x
        except Exception:
            pass

    # 4. Polish Global Best with Local Solver
    if best_global_norm < (norm(Fx_best) if Fx_best is not None else np.inf):
        x_best, Fx_best, _ = local_solver(best_global_x, best_global_Fx, max_iter=30)
    
    if Fx_best is not None and norm(Fx_best) <= 1e-8:
        return x_best

    return x_best

# EVOLVE-BLOCK-END
