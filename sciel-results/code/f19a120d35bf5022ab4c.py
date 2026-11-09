# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    The union volume is exactly separable: V = Vx * Vy, where Vx (resp. Vy) is the
    2-D area of the union of N sheared parallelograms in the (x, c) (resp. (y, c))
    plane.  We compute that area exactly via Gauss-Legendre quadrature (no Monte
    Carlo noise in the objective) and minimize it.

    Strategy (combining current + inspiration 1):
      1. Keich binary construction as the initial point.
      2. A low-dimensional Fourier parametric deformation field that maps grid
         indices to base coordinates, enforcing global coherence and reducing the
         effective search dimensionality (helps escape local traps).
      3. Nelder-Mead global optimization on the 16 base variables.
      4. Fine coordinate-descent polish with adaptive step size.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 585.0  # safety margin before 630s kill

    # --- Initial Keich construction ---
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    x_base = keich.copy()
    y_base = keich.copy()

    # --- Exact 2-D area of union of sheared rectangles in (coord, c) plane ---
    # Precompute Gauss-Legendre nodes and weights on [0,1]
    c_vals, w_vals = np.polynomial.legendre.leggauss(64)
    c_vals = 0.5 * (c_vals + 1.0)  # map from [-1,1] to [0,1]
    w_vals = 0.5 * w_vals          # scale weights
    idx_arange = np.arange(N, dtype=np.float64)
    interval_widths = (idx_arange + 1.0) / N  # (i+1)/N for i in 0..N-1

    def area_2d(base):
        """
        Compute the area of the union of N sheared rectangles (parallelograms)
        in the (coord, c) plane.  Rectangle i at height c occupies
        [base[i] + c*i/N, base[i] + (i+1)/N + c*i/N].
        We integrate the 1-D union length over c using Gauss-Legendre quadrature.
        """
        total_area = 0.0
        for c, w in zip(c_vals, w_vals):
            lefts = base + c * idx_arange / N
            rights = lefts + interval_widths

            # Merge intervals to compute union length
            order = np.argsort(lefts)
            lefts_sorted = lefts[order]
            rights_sorted = rights[order]

            total_len = 0.0
            cur_l = lefts_sorted[0]
            cur_r = rights_sorted[0]
            for k in range(1, N):
                if lefts_sorted[k] <= cur_r:
                    if rights_sorted[k] > cur_r:
                        cur_r = rights_sorted[k]
                else:
                    total_len += cur_r - cur_l
                    cur_l = lefts_sorted[k]
                    cur_r = rights_sorted[k]
            total_len += cur_r - cur_l

            total_area += w * total_len

        return total_area

    # --- Objective: minimize Vx * Vy ---
    def objective(params):
        x_opt = params[:N]
        y_opt = params[N:]
        vx = area_2d(x_opt)
        vy = area_2d(y_opt)
        if vx <= 0 or vy <= 0:
            return 1e10
        return vx * vy

    # --- Parametric deformation field: low-dimensional Fourier modes ---
    # This maps a small set of parameters to base positions, enforcing smoothness.
    # We use a combination of the Keich base and small perturbations.
    def parametric_deform(fourier_params, base):
        """
        Apply a small Fourier-mode deformation to the base positions.
        fourier_params: array of shape (4,) = [a1, b1, a2, b2]
        base: array of shape (N,)
        Returns deformed positions.
        """
        t = np.arange(N, dtype=np.float64) / N
        a1, b1, a2, b2 = fourier_params
        perturbation = a1 * np.sin(2 * np.pi * t) + b1 * np.cos(2 * np.pi * t) \
                     + a2 * np.sin(4 * np.pi * t) + b2 * np.cos(4 * np.pi * t)
        return base + perturbation

    # --- Objective with parametric deformation (for global coherence search) ---
    def objective_parametric(params):
        # First 4: x Fourier params, next 4: y Fourier params
        x_f = params[:4]
        y_f = params[4:8]
        x_opt = parametric_deform(x_f, x_base)
        y_opt = parametric_deform(y_f, y_base)
        vx = area_2d(x_opt)
        vy = area_2d(y_opt)
        if vx <= 0 or vy <= 0:
            return 1e10
        return vx * vy

    # --- Phase 1: Parametric global search (low-dim, smooth landscape) ---
    initial_param = np.zeros(8)
    best_param_obj = objective_parametric(initial_param)
    best_param = initial_param.copy()

    try:
        result_p = minimize(
            objective_parametric,
            initial_param,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True,
                'initial_simplex': 0.05 * np.ones(8)
            }
        )
        if result_p.fun < best_param_obj:
            best_param_obj = result_p.fun
            best_param = result_p.x
    except Exception:
        pass

    # Apply the best parametric deformation to get an improved starting point
    x_start = parametric_deform(best_param[:4], x_base)
    y_start = parametric_deform(best_param[4:8], y_base)

    # --- Phase 2: Full 16-variable Nelder-Mead optimization ---
    initial_params = np.concatenate([x_start, y_start])
    best_params = initial_params.copy()
    best_obj = objective(best_params)

    # Try multiple initial simplex sizes for robustness
    simplex_sizes = [0.1, 0.05, 0.2]
    for ss in simplex_sizes:
        if time.time() > deadline - 30.0:
            break
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True,
                    'initial_simplex': ss * np.ones(2 * N)
                }
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # --- Phase 3: Fine coordinate-descent polish ---
    if time.time() < deadline - 5.0:
        step = 1e-4
        no_improve = 0
        for iteration in range(200):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for idx in range(2 * N):
                old_val = best_params[idx]
                for delta in [step, -step, step/2, -step/2, step/4, -step/4]:
                    best_params[idx] = old_val + delta
                    obj = objective(best_params)
                    if obj < best_obj - 1e-14:
                        best_obj = obj
                        improved = True
                        break
                    best_params[idx] = old_val
                if not improved:
                    best_params[idx] = old_val
            if not improved:
                no_improve += 1
                step *= 0.7
                if step < 1e-7 or no_improve > 20:
                    break
            else:
                no_improve = 0

    # --- Extract final base positions ---
    x_final = best_params[:N]
    y_final = best_params[N:]

    # --- Build final positions array ---
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = x_final[i_flat]
    base_y = y_final[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
