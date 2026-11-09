"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for Bent Cigar function f12.
    Strategy:
    1. Brief global exploration using Sobol points to find a good starting basin.
    2. Local refinement using Powell's method (derivative-free, adaptive step sizes per dimension)
       which is effective for anisotropic functions like Bent Cigar.
    3. High-precision polish pass with Nelder-Mead.
    4. Coordinate-wise line search polish.
    5. Surrogate-assisted refinement using RBF interpolant and L-BFGS-B.
    6. Final coordinate-wise micro-polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit
        d = self.dim
        budget = self.budget

        state = {"count": 0, "ok": True, "best_x": None, "best_f": np.inf}

        def wf(x):
            if not state["ok"]:
                return 1e300
            if time.perf_counter() > deadline:
                state["ok"] = False
                return 1e300

            x = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x)):
                return 1e300

            try:
                v = float(func(x))
            except Exception:
                return 1e300

            state["count"] += 1
            if not np.isfinite(v):
                return 1e300

            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x.copy()

            return v

        # --- Phase 1: Global Exploration (Sobol) ---
        n_sobol = min(30, budget // 10)

        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_points = sampler.random(n_sobol)
            sobol_points = -5.0 + 10.0 * sobol_points

            best_idx = 0
            best_val = np.inf

            for i in range(n_sobol):
                if not state["ok"]:
                    break
                x = sobol_points[i]
                v = wf(x)
                if v < best_val:
                    best_val = v
                    best_idx = i

            x0 = sobol_points[best_idx].copy()
        except Exception:
            rng = np.random.default_rng(0)
            x0 = rng.uniform(-5, 5, d)
            v = wf(x0)
            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x0.copy()

        # --- Phase 2: Local Refinement (Powell) ---
        remaining_budget = budget - state["count"]

        if remaining_budget > 0 and state["ok"]:
            try:
                res = minimize(
                    wf,
                    x0,
                    method="Powell",
                    options={
                        "maxiter": 1000,
                        "maxfev": remaining_budget,
                        "xrtol": 1e-10,
                        "ftol": 1e-10,
                    },
                )
                xf = np.asarray(res.x, dtype=float)
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < state["best_f"]:
                        state["best_f"] = ff
                        state["best_x"] = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass

        # --- Phase 3: High-precision polish pass ---
        remaining_budget = budget - state["count"]
        if remaining_budget > 5 and state["ok"] and state["best_x"] is not None:
            try:
                x_start = state["best_x"].copy()

                res2 = minimize(
                    wf,
                    x_start,
                    method="Nelder-Mead",
                    options={
                        "maxiter": 200,
                        "maxfev": remaining_budget,
                        "xatol": 1e-12,
                        "fatol": 1e-12,
                        "initial_simplex": x_start + 1e-4 * np.eye(d),
                    },
                )
                xf2 = np.asarray(res2.x, dtype=float)
                ff2 = res2.fun
                if np.all(np.isfinite(xf2)) and np.isfinite(ff2):
                    if ff2 < state["best_f"]:
                        state["best_f"] = ff2
                        state["best_x"] = np.clip(xf2, -5.0, 5.0)
            except Exception:
                pass

        # --- Phase 4: Coordinate-wise line search polish ---
        remaining_budget = budget - state["count"]
        if remaining_budget > 2 and state["ok"] and state["best_x"] is not None:
            try:
                x_best = state["best_x"].copy()
                f_best = state["best_f"]

                for _ in range(min(remaining_budget // d, 10)):
                    if not state["ok"] or time.perf_counter() > deadline:
                        break
                    improved = False
                    for j in range(d):
                        if not state["ok"] or time.perf_counter() > deadline:
                            break
                        for delta in [1e-3, 1e-4, 1e-5, -1e-3, -1e-4, -1e-5]:
                            if not state["ok"] or time.perf_counter() > deadline:
                                break
                            x_try = x_best.copy()
                            x_try[j] += delta
                            x_try = np.clip(x_try, -5.0, 5.0)
                            v_try = wf(x_try)
                            if v_try < f_best - 1e-15:
                                x_best = x_try
                                f_best = v_try
                                improved = True
                                break
                    if not improved:
                        break

                if f_best < state["best_f"]:
                    state["best_f"] = f_best
                    state["best_x"] = x_best
            except Exception:
                pass

        # --- Phase 5: Surrogate-assisted refinement ---
        # Fit an RBF interpolant to recent points and use L-BFGS-B to navigate
        remaining_budget = budget - state["count"]
        if remaining_budget > 20 and state["ok"] and state["best_x"] is not None:
            try:
                # Collect points from the optimization history
                # We'll generate a small set of points around the current best
                # to fit a local surrogate
                
                x_center = state["best_x"].copy()
                f_center = state["best_f"]
                
                # Generate points in a local neighborhood for surrogate fitting
                # Use a small number of points to keep it fast
                n_local = min(30, remaining_budget // 3)
                
                if n_local > 5:
                    # Create local points
                    rng_local = np.random.default_rng(123)
                    local_points = []
                    local_values = []
                    
                    # Start with the center
                    local_points.append(x_center.copy())
                    local_values.append(f_center)
                    
                    # Add points in a local neighborhood
                    for _ in range(n_local - 1):
                        if not state["ok"] or time.perf_counter() > deadline:
                            break
                        # Random point in a small box around center
                        radius = 0.5
                        x_try = x_center + rng_local.uniform(-radius, radius, d)
                        x_try = np.clip(x_try, -5.0, 5.0)
                        v_try = wf(x_try)
                        if np.isfinite(v_try):
                            local_points.append(x_try.copy())
                            local_values.append(v_try)
                    
                    if len(local_points) >= d + 2:
                        local_points = np.array(local_points)
                        local_values = np.array(local_values)
                        
                        # Fit RBF interpolant
                        # Use multiquadric RBF: phi(r) = sqrt(1 + (eps*r)^2)
                        eps = 1.0  # Shape parameter
                        
                        def rbf_kernel(X, Y, eps=1.0):
                            """Compute RBF kernel matrix between X and Y"""
                            # X: (n, d), Y: (m, d)
                            # Returns: (n, m)
                            n = X.shape[0]
                            m = Y.shape[0]
                            # Compute pairwise distances
                            # Using broadcasting
                            X2 = np.sum(X**2, axis=1).reshape(-1, 1)
                            Y2 = np.sum(Y**2, axis=1).reshape(1, -1)
                            D2 = X2 + Y2 - 2 * X @ Y.T
                            D2 = np.maximum(D2, 0)  # Numerical stability
                            D = np.sqrt(D2)
                            return np.sqrt(1.0 + (eps * D)**2)
                        
                        # Solve for RBF weights
                        K = rbf_kernel(local_points, local_points, eps)
                        # Add small regularization for numerical stability
                        K += 1e-8 * np.eye(K.shape[0])
                        
                        try:
                            weights = np.linalg.solve(K, local_values)
                        except np.linalg.LinAlgError:
                            weights = np.linalg.lstsq(K, local_values, rcond=None)[0]
                        
                        # Define surrogate function and gradient
                        def surrogate(x):
                            x = np.asarray(x, dtype=float)
                            Kx = rbf_kernel(x.reshape(1, -1), local_points, eps).flatten()
                            return float(np.dot(Kx, weights))
                        
                        def surrogate_grad(x):
                            x = np.asarray(x, dtype=float)
                            # Gradient of RBF: d/dx sqrt(1 + (eps*||x - x_i||^2))
                            # = eps^2 * (x - x_i) / sqrt(1 + (eps*||x - x_i||^2))
                            # = eps^2 * (x - x_i) / phi(||x - x_i||)
                            grad = np.zeros(d)
                            for i in range(len(local_points)):
                                diff = x - local_points[i]
                                dist_sq = np.dot(diff, diff)
                                phi_val = np.sqrt(1.0 + (eps * np.sqrt(dist_sq))**2)
                                # d phi / dx = eps^2 * diff / phi
                                grad += weights[i] * (eps**2 * diff / phi_val)
                            return grad
                        
                        # Use L-BFGS-B to optimize the surrogate
                        res_surr = minimize(
                            surrogate,
                            x_center,
                            jac=surrogate_grad,
                            method='L-BFGS-B',
                            bounds=[(-5.0, 5.0)] * d,
                            options={
                                'maxiter': 50,
                                'maxfun': 100,
                                'gtol': 1e-8,
                                'ftol': 1e-10,
                            }
                        )
                        
                        if res_surr.success and np.all(np.isfinite(res_surr.x)):
                            x_surr = np.clip(res_surr.x, -5.0, 5.0)
                            # Evaluate the true function at the surrogate optimum
                            v_surr = wf(x_surr)
                            if v_surr < state["best_f"]:
                                state["best_f"] = v_surr
                                state["best_x"] = x_surr.copy()
            except Exception:
                pass

        # --- Phase 6: Final coordinate-wise micro-polish ---
        remaining_budget = budget - state["count"]
        if remaining_budget > 2 and state["ok"] and state["best_x"] is not None:
            try:
                x_best = state["best_x"].copy()
                f_best = state["best_f"]

                for _ in range(min(remaining_budget // d, 5)):
                    if not state["ok"] or time.perf_counter() > deadline:
                        break
                    improved = False
                    for j in range(d):
                        if not state["ok"] or time.perf_counter() > deadline:
                            break
                        for delta in [1e-5, 1e-6, 1e-7, -1e-5, -1e-6, -1e-7]:
                            if not state["ok"] or time.perf_counter() > deadline:
                                break
                            x_try = x_best.copy()
                            x_try[j] += delta
                            x_try = np.clip(x_try, -5.0, 5.0)
                            v_try = wf(x_try)
                            if v_try < f_best - 1e-15:
                                x_best = x_try
                                f_best = v_try
                                improved = True
                                break
                    if not improved:
                        break

                if f_best < state["best_f"]:
                    state["best_f"] = f_best
                    state["best_x"] = x_best
            except Exception:
                pass

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
