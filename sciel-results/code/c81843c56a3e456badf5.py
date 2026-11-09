"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using a multi-strategy approach.
    
    Strategy:
    1. Try SciPy fsolve (Levenberg-Marquardt) with tight tolerance.
    2. Try Broyden's method with line search from best point.
    3. Try scipy.optimize.root with different methods.
    4. Random restarts with Newton if still not converged.
    
    The key insight is to use multiple robust solvers and track the best
    point found across all attempts.
    """
    start_time = time.time()
    DEADLINE = 130.0  # Safety margin before 150s hard limit

    def time_ok():
        return time.time() - start_time < DEADLINE

    x = np.array(x0, dtype=float).copy()
    n = 2

    def compute_F(val):
        val = np.asarray(val, dtype=float)
        return np.asarray(F(val), dtype=float)

    best_x = x.copy()
    best_norm = np.inf

    # First, evaluate F at x0 to have a baseline
    try:
        Fx0 = compute_F(x)
        if np.all(np.isfinite(Fx0)):
            best_norm = np.linalg.norm(Fx0)
        else:
            return x
    except Exception:
        return x

    if best_norm <= 1e-8:
        return x

    # Method 1: SciPy fsolve (Levenberg-Marquardt)
    if time_ok():
        try:
            sol, info, ier, msg = so.fsolve(
                lambda x: np.asarray(F(x), dtype=float), 
                x, 
                full_output=True, 
                maxfev=200, 
                xtol=1e-14,
                factor=0.1
            )
            if np.all(np.isfinite(sol)):
                norm_sol = np.linalg.norm(compute_F(sol))
                if np.isfinite(norm_sol) and norm_sol < best_norm:
                    best_norm = norm_sol
                    best_x = sol.copy()
                if best_norm <= 1e-8:
                    return best_x
        except Exception:
            pass

    # Method 2: Try scipy.optimize.root with different methods
    if time_ok() and best_norm > 1e-8:
        for method in ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson']:
            if not time_ok():
                break
            try:
                res = so.root(
                    lambda x: np.asarray(F(x), dtype=float),
                    best_x,
                    method=method,
                    tol=1e-14,
                    options={'maxiter': 100}
                )
                if res.success and np.all(np.isfinite(res.x)):
                    norm_sol = np.linalg.norm(compute_F(res.x))
                    if np.isfinite(norm_sol) and norm_sol < best_norm:
                        best_norm = norm_sol
                        best_x = res.x.copy()
                    if best_norm <= 1e-8:
                        return best_x
            except Exception:
                pass

    # Method 3: Broyden's Method with line search
    if time_ok() and best_norm > 1e-8:
        x_broyden = best_x.copy()
        # Initial Jacobian using central differences
        h_init = 1e-8 * max(1.0, np.linalg.norm(x_broyden))
        try:
            J = np.zeros((2, 2))
            Fx = compute_F(x_broyden)
            for j in range(2):
                e = np.zeros(2)
                e[j] = h_init
                J[:, j] = (compute_F(x_broyden + e) - compute_F(x_broyden - e)) / (2 * h_init)
        except Exception:
            J = None
            Fx = compute_F(x_broyden)
        
        if J is not None and np.all(np.isfinite(J)):
            norm_Fx = np.linalg.norm(Fx)
            if np.isfinite(norm_Fx) and norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x_broyden.copy()
            
            max_iters = 40
            for it in range(max_iters):
                if not time_ok():
                    break
                if norm_Fx <= 1e-10:
                    break
                
                # Solve J @ dx = -Fx
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    try:
                        dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                    except:
                        break
                
                # Backtracking Line Search
                alpha = 1.0
                accepted = False
                for _ in range(25):
                    x_trial = x_broyden + alpha * dx
                    try:
                        Fx_trial = compute_F(x_trial)
                        norm_Fx_trial = np.linalg.norm(Fx_trial)
                    except:
                        alpha *= 0.5
                        continue
                    if np.isfinite(norm_Fx_trial) and norm_Fx_trial < norm_Fx:
                        x_new = x_trial
                        Fx_new = Fx_trial
                        norm_Fx_new = norm_Fx_trial
                        s = x_new - x_broyden
                        y = Fx_new - Fx
                        accepted = True
                        break
                    alpha *= 0.5
                
                if not accepted:
                    break
                
                if norm_Fx_new < best_norm:
                    best_norm = norm_Fx_new
                    best_x = x_new.copy()
                
                # Update Jacobian
                if np.linalg.norm(s) > 1e-14:
                    J = J + np.outer(y - J @ s, s) / np.dot(s, s)
                else:
                    h_re = 1e-8 * max(1.0, np.linalg.norm(x_new))
                    try:
                        J = np.zeros((2, 2))
                        Fx_new = compute_F(x_new)
                        for j in range(2):
                            e = np.zeros(2)
                            e[j] = h_re
                            J[:, j] = (compute_F(x_new + e) - compute_F(x_new - e)) / (2 * h_re)
                    except:
                        break
                
                x_broyden = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new

    # Check if Broyden converged
    if best_norm <= 1e-8:
        return best_x

    # Method 4: Random restarts with Newton
    if best_norm > 1e-8 and time_ok():
        np.random.seed(42)
        for attempt in range(15):
            if not time_ok():
                break
            # Perturb from best_x with varying scales
            scale = 0.3 * (attempt + 1)
            x_try = best_x + np.random.randn(2) * scale
            x_cur = x_try.copy()
            
            for it in range(25):
                if not time_ok():
                    break
                try:
                    Fx = compute_F(x_cur)
                    norm_Fx = np.linalg.norm(Fx)
                except:
                    break
                if not np.isfinite(norm_Fx):
                    break
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x_cur.copy()
                if norm_Fx <= 1e-10:
                    break
                
                h = 1e-7 * max(1.0, np.linalg.norm(x_cur))
                Fx0 = Fx
                e1 = np.array([h, 0.0])
                e2 = np.array([0.0, h])
                J = np.zeros((2, 2))
                try:
                    J[:, 0] = (compute_F(x_cur + e1) - Fx0) / h
                    J[:, 1] = (compute_F(x_cur + e2) - Fx0) / h
                except:
                    break
                
                if not np.all(np.isfinite(J)):
                    break
                
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                
                if not np.all(np.isfinite(dx)):
                    break
                
                # Line search
                alpha = 1.0
                accepted = False
                for _ in range(15):
                    x_new = x_cur + alpha * dx
                    try:
                        Fx_new = compute_F(x_new)
                        norm_Fx_new = np.linalg.norm(Fx_new)
                    except:
                        alpha *= 0.5
                        continue
                    if np.isfinite(norm_Fx_new) and norm_Fx_new < norm_Fx:
                        x_cur = x_new
                        accepted = True
                        break
                    alpha *= 0.5
                
                if not accepted:
                    break

    # Final check
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)

# EVOLVE-BLOCK-END
