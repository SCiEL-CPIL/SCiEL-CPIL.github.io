"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def find_root_system(F, x0):
    """
    Robust root finder for 2D systems.
    
    Strategy:
    1. Direct SciPy fsolve (Levenberg-Marquardt) with tight tolerance.
       This is a robust, C-optimized hybrid method that handles singular Jacobians
       and non-convergence well. It uses finite differences internally but is
       highly efficient for 2D systems.
    2. If fsolve fails to converge within budget, try Broyden's method with
       finite-difference initial Jacobian and line search.
    3. Random restarts with Newton if still not converged.
    
    Rationale: The previous approach started with Broyden which can be slower
    to converge or get stuck. Starting with the most robust solver (fsolve/LM)
    often finds the root faster with fewer calls for well-behaved 2D systems.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s hard limit

    def time_ok():
        return time.time() - start_time < DEADLINE

    x = np.array(x0, dtype=float).copy()
    n = 2

    def compute_F(val):
        return np.asarray(F(val), dtype=float)

    best_x = x.copy()
    best_norm = np.inf

    # First, evaluate F at x0 to have a baseline
    try:
        Fx0 = compute_F(x)
        best_norm = np.linalg.norm(Fx0)
    except Exception:
        return x

    if best_norm <= 1e-8:
        return x

    # Method 1: SciPy fsolve (Levenberg-Marquardt)
    # This is often the most robust and efficient for small 2D systems.
    if time_ok():
        try:
            # We give it a generous budget but not the full 500, leaving room for fallbacks.
            # fsolve counts function evaluations. For 2D, it typically needs < 20 evals.
            sol, info, ier, msg = so.fsolve(
                lambda x: np.asarray(F(x), dtype=float), 
                x, 
                full_output=True, 
                maxfev=300, 
                xtol=1e-12,
                factor=0.1
            )
            if np.all(np.isfinite(sol)):
                norm_sol = np.linalg.norm(compute_F(sol))
                if norm_sol < best_norm:
                    best_norm = norm_sol
                    best_x = sol.copy()
                if best_norm <= 1e-8:
                    return best_x
        except Exception:
            pass

    # Method 2: Broyden's Method fallback
    # If fsolve didn't converge, try Broyden from the best point found.
    if time_ok() and best_norm > 1e-8:
        x_broyden = best_x.copy()
        # Initial Jacobian using central differences (5 calls)
        h_init = 1e-7 * max(1.0, np.linalg.norm(x_broyden))
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
        
        if J is not None:
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x_broyden.copy()
            
            max_iters = 50
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
                x_trial = x_broyden + alpha * dx
                Fx_trial = compute_F(x_trial)
                norm_Fx_trial = np.linalg.norm(Fx_trial)
                
                if norm_Fx_trial < norm_Fx:
                    x_new = x_trial
                    Fx_new = Fx_trial
                    norm_Fx_new = norm_Fx_trial
                    s = x_new - x_broyden
                    y = Fx_new - Fx
                else:
                    accepted = False
                    for _ in range(20):
                        alpha *= 0.5
                        x_trial = x_broyden + alpha * dx
                        Fx_trial = compute_F(x_trial)
                        norm_Fx_trial = np.linalg.norm(Fx_trial)
                        if norm_Fx_trial < norm_Fx:
                            x_new = x_trial
                            Fx_new = Fx_trial
                            norm_Fx_new = norm_Fx_trial
                            s = x_new - x_broyden
                            y = Fx_new - Fx
                            accepted = True
                            break
                    if not accepted:
                        break
                
                if norm_Fx_new < best_norm:
                    best_norm = norm_Fx_new
                    best_x = x_new.copy()
                
                if np.linalg.norm(s) > 1e-12:
                    J = J + np.outer(y - J @ s, s) / np.dot(s, s)
                else:
                    h_re = 1e-7 * max(1.0, np.linalg.norm(x_new))
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

    # Method 3: Random restarts with simple Newton
    if best_norm > 1e-8 and time_ok():
        for attempt in range(10):
            if not time_ok():
                break
            # Perturb from best_x
            x_try = best_x + np.random.randn(2) * 0.5
            x_cur = x_try.copy()
            
            for it in range(30):
                if not time_ok():
                    break
                Fx = compute_F(x_cur)
                norm_Fx = np.linalg.norm(Fx)
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
                J[:, 0] = (compute_F(x_cur + e1) - Fx0) / h
                J[:, 1] = (compute_F(x_cur + e2) - Fx0) / h
                
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                
                alpha = 1.0
                x_new = x_cur + alpha * dx
                Fx_new = compute_F(x_new)
                if np.linalg.norm(Fx_new) < norm_Fx:
                    x_cur = x_new
                else:
                    alpha = 0.5
                    x_new = x_cur + alpha * dx
                    Fx_new = compute_F(x_new)
                    if np.linalg.norm(Fx_new) < norm_Fx:
                        x_cur = x_new
                    else:
                        break

    # Final check
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)

# EVOLVE-BLOCK-END
