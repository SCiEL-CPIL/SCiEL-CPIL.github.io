"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5.

    Structural idea: parameterize the cell as a smooth superposition of C4v-symmetric
    Fourier modes (2,0),(0,2),(2,2),(4,0),(0,2) with a tunable threshold, enforced to
    C4v by averaging over all 8 symmetry operations. Refinement machinery: coarse grid
    -> fine grid -> Nelder-Mead local search (with a smooth log-sum-exp surrogate) ->
    pixel refinement. A fast 289-plane-wave Galerkin evaluator scores every candidate.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    t_start = time.time()
    DEADLINE = 218.0  # safety margin before 230s wall-clock kill

    def time_ok():
        return time.time() - t_start < DEADLINE

    # ---- Plane-wave setup: 17x17 = 289 plane waves (matches evaluator) ----
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289

    # ---- 30 k-points over the irreducible BZ ----
    pi = np.pi
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)
    nk = len(k_points)

    # Precompute (k+G)^2 for each k-point
    k2_all = np.zeros((nk, n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2

    # Precompute difference indices for Fourier coupling (sparse structure)
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            idx_x = int(dGx) + half
            idx_y = int(dGy) + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)

    # Precompute the (k+G)^2 contribution on the valid (sparse) entries per k-point
    k2_valid = np.zeros((nk, len(valid_rows)))
    for i in range(nk):
        k2_valid[i] = k2_all[i, valid_rows]

    # ---- Coordinate grid for cell generation ----
    coords = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(coords, coords, indexing='xy')

    def enforce_symmetry(cell):
        """Enforce full C4v (square symmetry group) about the cell centre."""
        cell = np.array(cell, dtype=int)
        sym = [
            cell,
            np.rot90(cell, 1),
            np.rot90(cell, 2),
            np.rot90(cell, 3),
            cell[:, ::-1].copy(),
            cell[::-1, :].copy(),
            cell.T.copy(),
            np.rot90(cell.T, 1),
        ]
        avg = np.sum(sym, axis=0)
        return (avg >= 4).astype(int)

    def get_fourier_coeffs(cell):
        """Fourier coefficients of 1/eps via FFT (exact pixel-map coefficients)."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.empty(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            coeffs[idx] = fft_2d[gx % size, gy % size] / (size * size)
        return coeffs

    def compute_band_ratio(coeffs):
        """Exact gap-midgap ratio 2(w5min - w4max)/(w5min + w4max) over the BZ."""
        max_band4 = -np.inf
        min_band5 = np.inf
        for k_idx in range(nk):
            H_vals = k2_valid[k_idx] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) * 0.5
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                if eigvals[3] > max_band4:
                    max_band4 = eigvals[3]
                if eigvals[4] < min_band5:
                    min_band5 = eigvals[4]
        if not np.isfinite(min_band5) or not np.isfinite(max_band4):
            return -np.inf
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def score_cell(cell):
        coeffs = get_fourier_coeffs(cell)
        return compute_band_ratio(coeffs)

    def generate_cell(p):
        """
        p = (offset, A20, A22, A40, A02, phase22)
        field = offset
              + A20*(cos4piX + cos4piY)
              + A22*(cos(4pi(X+Y)+ph) + cos(4pi(X-Y)+ph))
              + A40*cos8piX + A02*cos8piY
        cell = field > 0, then C4v symmetrized.
        """
        offset, A20, A22, A40, A02, phase22 = p
        field = np.full((size, size), offset)
        if A20 != 0.0:
            field += A20 * (np.cos(4 * pi * X) + np.cos(4 * pi * Y))
        if A22 != 0.0:
            field += A22 * (np.cos(4 * pi * (X + Y) + phase22) +
                            np.cos(4 * pi * (X - Y) + phase22))
        if A40 != 0.0:
            field += A40 * np.cos(8 * pi * X)
        if A02 != 0.0:
            field += A02 * np.cos(8 * pi * Y)
        cell_raw = (field > 0.0).astype(int)
        return enforce_symmetry(cell_raw)

    def fill_frac(cell):
        return np.sum(cell) / (size * size)

    def smooth_surrogate(coeffs, beta=40.0):
        """
        Smooth surrogate of the gap-midgap ratio using log-sum-exp.
        soft-max over k of w4(k)  ~  (1/beta) log sum exp(beta w4)
        soft-min over k of w5(k)  ~  -(1/beta) log sum exp(-beta w5)
        As beta -> inf this converges to the exact max/min; finite beta is smooth
        and gives a usable descent direction for the local (Nelder-Mead) stage.
        """
        w4 = np.empty(nk)
        w5 = np.empty(nk)
        for k_idx in range(nk):
            H_vals = k2_valid[k_idx] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) * 0.5
            ev = np.sort(np.linalg.eigvalsh(H))
            if len(ev) >= 5:
                w4[k_idx] = ev[3]
                w5[k_idx] = ev[4]
            else:
                w4[k_idx] = -1e9
                w5[k_idx] = 1e9
        # soft-max of w4
        m4 = np.max(w4)
        logsum4 = m4 + np.log(np.sum(np.exp(beta * (w4 - m4))))
        soft_max4 = logsum4 / beta
        # soft-min of w5
        m5 = np.min(w5)
        logsum5 = -m5 + np.log(np.sum(np.exp(-beta * (w5 - m5))))
        soft_min5 = -logsum5 / beta
        gap = soft_min5 - soft_max4
        midgap = (soft_min5 + soft_max4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    # ==================================================================
    # Phase 1: Coarse grid search over Fourier parameters
    # ==================================================================
    best_ratio = -np.inf
    best_cell = None
    best_params = None

    # Sensible priors: moderate fill fraction, significant (2,0)/(2,2) content
    offset_coarse = [0.0, 0.1, 0.2, 0.3, 0.4]
    A20_coarse = [0.0, 0.3, 0.6, 0.9, 1.2]
    A22_coarse = [0.0, 0.3, 0.6, 0.9, 1.2]
    A40_coarse = [0.0, 0.3, 0.6]
    A02_coarse = [0.0, 0.3, 0.6]

    n_eval = 0
    for offset in offset_coarse:
        for A20 in A20_coarse:
            for A22 in A22_coarse:
                for A40 in A40_coarse:
                    for A02 in A02_coarse:
                        if not time_ok():
                            break
                        p = (offset, A20, A22, A40, A02, 0.0)
                        cell = generate_cell(p)
                        f = fill_frac(cell)
                        if f < 0.15 or f > 0.85:
                            continue
                        r = score_cell(cell)
                        n_eval += 1
                        if r > best_ratio:
                            best_ratio = r
                            best_cell = cell.copy()
                            best_params = p
                    if not time_ok():
                        break
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    if not time_ok():
        break_outer = True
    else:
        break_outer = False

    if best_cell is None:
        # Fallback: a known-reasonable configuration
        best_params = (0.2, 0.6, 0.6, 0.3, 0.3, 0.0)
        best_cell = generate_cell(best_params)
        best_ratio = score_cell(best_cell)

    # ==================================================================
    # Phase 2: Fine grid around the best coarse candidate
    # ==================================================================
    if not break_outer and time_ok() and best_params is not None:
        bo = best_params
        def fine_grid(base, lo, hi, n=5):
            return np.linspace(max(lo, base - 0.25), min(hi, base + 0.25), n)
        f_offset = fine_grid(bo[0], -0.5, 0.9)
        f_A20 = fine_grid(bo[1], 0.0, 1.8)
        f_A22 = fine_grid(bo[2], 0.0, 1.8)
        f_A40 = fine_grid(bo[3], 0.0, 1.2)
        f_A02 = fine_grid(bo[4], 0.0, 1.2)
        f_ph22 = np.linspace(-0.6, 0.6, 3)

        for offset in f_offset:
            for A20 in f_A20:
                for A22 in f_A22:
                    for A40 in f_A40:
                        for A02 in f_A02:
                            for ph22 in f_ph22:
                                if not time_ok():
                                    break
                                p = (offset, A20, A22, A40, A02, ph22)
                                cell = generate_cell(p)
                                f = fill_frac(cell)
                                if f < 0.15 or f > 0.85:
                                    continue
                                r = score_cell(cell)
                                if r > best_ratio:
                                    best_ratio = r
                                    best_cell = cell.copy()
                                    best_params = p
                            if not time_ok():
                                break
                        if not time_ok():
                            break
                    if not time_ok():
                        break
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break_outer = True

    # ==================================================================
    # Phase 3: Nelder-Mead local search on the 6 Fourier parameters,
    #          using the smooth log-sum-exp surrogate as the objective.
    # ==================================================================
    if not break_outer and time_ok() and best_params is not None:
        def obj(p):
            cell = generate_cell(p)
            f = fill_frac(cell)
            if f < 0.12 or f > 0.88:
                return -1e6
            coeffs = get_fourier_coeffs(cell)
            return smooth_surrogate(coeffs, beta=40.0)

        # Nelder-Mead
        def nelder_mead(f, x0, bounds, max_iter=600, time_budget=60.0):
            x = np.array(x0, dtype=float)
            n = len(x)
            simplex = [x.copy()]
            for i in range(n):
                xi = x.copy()
                step = 0.1 * (bounds[i][1] - bounds[i][0]) + 1e-3
                xi[i] += step
                simplex.append(xi)
            fvals = [f(s) for s in simplex]
            t0 = time.time()
            for it in range(max_iter):
                if time.time() - t0 > time_budget:
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                centroid = np.mean(simplex[:-1], axis=0)
                # reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                fr = f(xr)
                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[0]:
                    # expansion
                    xe = centroid + 2.0 * (xr - centroid)
                    fe = f(xe)
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                else:
                    # contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = f(xc)
                    if fc < fvals[-1]:
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = f(simplex[i])
            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        bounds = [(-0.5, 0.9), (0.0, 1.8), (0.0, 1.8), (0.0, 1.2), (0.0, 1.2), (-0.8, 0.8)]
        x0 = list(best_params)
        # clip initial point into bounds
        x0 = [min(max(x0[i], bounds[i][0]), bounds[i][1]) for i in range(6)]

        # Local time budget: leave ~30s for pixel refinement + finalization
        remaining = DEADLINE - (time.time() - t_start)
        nm_budget = max(10.0, min(80.0, remaining - 35.0))
        x_opt, _ = nelder_mead(obj, x0, bounds, max_iter=800, time_budget=nm_budget)

        # Clip and evaluate the optimizer's result with the EXACT objective
        x_opt = [min(max(x_opt[i], bounds[i][0]), bounds[i][1]) for i in range(6)]
        cell_opt = generate_cell(tuple(x_opt))
        f_opt = fill_frac(cell_opt)
        if 0.12 <= f_opt <= 0.88:
            r_opt = score_cell(cell_opt)
            if r_opt > best_ratio:
                best_ratio = r_opt
                best_cell = cell_opt
                best_params = tuple(x_opt)

    # ==================================================================
    # Phase 4: Pixel-level refinement (independent quadrant pixels)
    # ==================================================================
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 25:
        current_cell = best_cell.copy()
        current_ratio = best_ratio

        indep_pixels = [(i, j) for i in range(16) for j in range(16)]
        np.random.shuffle(indep_pixels)

        refine_start = time.time()
        refine_limit = min(20.0, max(5.0, DEADLINE - (time.time() - t_start) - 12.0))

        no_improve = 0
        for (i, j) in indep_pixels:
            if time.time() - refine_start > refine_limit or not time_ok():
                break
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            if np.array_equal(new_cell, current_cell):
                continue
            r = score_cell(new_cell)
            if r > current_ratio + 1e-9:
                current_cell = new_cell
                current_ratio = r
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 60:
                    break

        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_cell = current_cell

    # Final safety: ensure we return a valid, symmetric cell
    best_cell = enforce_symmetry(best_cell)
    return best_cell.tolist()
# EVOLVE-BLOCK-END
