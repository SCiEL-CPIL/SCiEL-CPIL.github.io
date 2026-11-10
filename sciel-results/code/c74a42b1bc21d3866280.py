# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set with N=8 (64 tubes).
    Returns base positions (x_k, y_k) for tube k = (i,j) with i=k//8, j=k%8.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin

    N = 8
    pos = int(np.log2(N))  # 3

    # ------------------------------------------------------------------
    # Keich binary construction for 1-D N = 2^pos base offsets
    # The optimal offsets for the 1-D Kakeya problem with N=2^pos tubes
    # of width 1/N sheared by i/N.
    # ------------------------------------------------------------------
    def keich_offsets(N, pos):
        """
        Compute Keich offsets for N = 2^pos tubes.
        Tube i occupies [a_i + c*i/N, a_i + c*i/N + 1/N] at height c.
        The union measure is minimized by specific offsets.
        """
        # For N=2^pos, the Keich construction gives:
        # a_i = sum_{k=1}^{pos} (1 - i_k) * (2^{-k}) / pos
        # where i_k is the k-th bit of i (from LSB, k=1 is LSB)
        j_terms = np.arange(1, pos + 1)
        ks = np.empty(N, dtype=np.float64)
        for idx in range(N):
            # Get binary representation with pos bits
            bits = [(idx >> k) & 1 for k in range(pos)]
            # bits[k] is the (k+1)-th bit, k=0 is LSB
            # j_terms[k] = k+1
            eps = np.array(bits, dtype=float)
            ks[idx] = np.sum((1 - eps) * (2.0 ** (-j_terms)) / pos)
        return ks

    keich_base = keich_offsets(N, pos)

    # ------------------------------------------------------------------
    # Fast volume estimator using vectorized operations
    # V = integral_c Ux(c) * Uy(c) dc
    # where Ux(c) = measure(union_i [a_i + c*i/N, a_i + c*i/N + 1/N])
    # ------------------------------------------------------------------
    def est_volume_2d(offs_x, offs_y, N, num_c=100, num_xy=1000):
        """
        Estimate V = integral_c A(c) dc where A(c) = Ux(c) * Uy(c)
        using grid-based approximation.
        """
        i_vals = np.arange(N)
        c_grid = np.linspace(0.0, 1.0, num_c + 1)

        # x range: min offset to max(offset + i/N + 1/N)
        x_min = np.min(offs_x)
        x_max = np.max(offs_x + i_vals / N + 1.0 / N)
        y_min = np.min(offs_y)
        y_max = np.max(offs_y + i_vals / N + 1.0 / N)

        if x_max - x_min <= 1e-12 or y_max - y_min <= 1e-12:
            return 0.0

        x_grid = np.linspace(x_min, x_max, num_xy)
        y_grid = np.linspace(y_min, y_max, num_xy)
        dx = (x_max - x_min) / (num_xy - 1)
        dy = (y_max - y_min) / (num_xy - 1)

        total = 0.0
        dc = 1.0 / num_c

        for c in c_grid[:-1]:
            # x coverage: union of intervals [offs_x[i] + c*i/N, offs_x[i] + c*i/N + 1/N]
            # For each x_grid point, check if it's in any interval
            # Vectorized: for each i, create mask
            x_cov = np.zeros(num_xy, dtype=bool)
            for i in range(N):
                lo_x = offs_x[i] + c * i / N
                hi_x = lo_x + 1.0 / N
                x_cov |= (x_grid >= lo_x) & (x_grid <= hi_x)
            Ux = np.sum(x_cov) * dx

            y_cov = np.zeros(num_xy, dtype=bool)
            for j in range(N):
                lo_y = offs_y[j] + c * j / N
                hi_y = lo_y + 1.0 / N
                y_cov |= (y_grid >= lo_y) & (y_grid <= hi_y)
            Uy = np.sum(y_cov) * dy

            total += Ux * Uy

        V = total * dc
        return V

    # ------------------------------------------------------------------
    # Even faster: use interval union directly
    # ------------------------------------------------------------------
    def _union_measure(intervals, lo_bound, hi_bound, num_grid=500):
        """
        Compute measure of union of intervals within [lo_bound, hi_bound]
        using grid sampling.
        """
        grid = np.linspace(lo_bound, hi_bound, num_grid)
        dgrid = (hi_bound - lo_bound) / (num_grid - 1)
        cov = np.zeros(num_grid, dtype=bool)
        for lo, hi in intervals:
            cov |= (grid >= lo) & (grid <= hi)
        return np.sum(cov) * dgrid

    def est_volume_2d_fast(offs_x, offs_y, N, num_c=50, num_grid=500):
        """
        Faster volume estimate using interval union.
        """
        i_vals = np.arange(N)
        c_grid = np.linspace(0.0, 1.0, num_c + 1)

        x_min = np.min(offs_x)
        x_max = np.max(offs_x + i_vals / N + 1.0 / N)
        y_min = np.min(offs_y)
        y_max = np.max(offs_y + i_vals / N + 1.0 / N)

        if x_max - x_min <= 1e-12 or y_max - y_min <= 1e-12:
            return 0.0

        total = 0.0
        dc = 1.0 / num_c

        for c in c_grid[:-1]:
            # Build x intervals
            x_intervals = [(offs_x[i] + c * i / N, offs_x[i] + c * i / N + 1.0 / N) for i in range(N)]
            Ux = _union_measure(x_intervals, x_min, x_max, num_grid)

            y_intervals = [(offs_y[j] + c * j / N, offs_y[j] + c * j / N + 1.0 / N) for j in range(N)]
            Uy = _union_measure(y_intervals, y_min, y_max, num_grid)

            total += Ux * Uy

        V = total * dc
        return V

    # ------------------------------------------------------------------
    # Try several offset configurations
    # ------------------------------------------------------------------
    best_V = np.inf
    best_x = keich_base.copy()
    best_y = keich_base.copy()

    # Candidate offset sets
    candidates = [keich_base]

    # Try reversed Keich
    candidates.append(keich_base[::-1].copy())

    # Try sorted Keich
    candidates.append(np.sort(keich_base))

    # Try uniform spacing
    candidates.append(np.linspace(0, 0.5, N))

    # Try small perturbations
    rng_local = np.random.default_rng(42)
    for _ in range(10):
        cand = keich_base + rng_local.uniform(-0.03, 0.03, size=N)
        candidates.append(cand)

    # Try scaling
    for s in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]:
        candidates.append(keich_base * s)

    # Try translation
    for t in [-0.05, -0.02, 0.0, 0.02, 0.05]:
        candidates.append(keich_base + t)

    # Evaluate combinations (limit to keep within time)
    num_c_eval = 30
    num_grid_eval = 400
    max_evals = 200
    eval_count = 0

    for ox in candidates:
        for oy in candidates:
            if eval_count >= max_evals:
                break
            if time.time() > deadline - 60:
                break
            V = est_volume_2d_fast(ox, oy, N, num_c=num_c_eval, num_grid=num_grid_eval)
            if V < best_V:
                best_V = V
                best_x = ox.copy()
                best_y = oy.copy()
            eval_count += 1
        if eval_count >= max_evals or time.time() > deadline - 60:
            break

    # ------------------------------------------------------------------
    # Local refinement using scipy if available
    # ------------------------------------------------------------------
    try:
        from scipy.optimize import minimize
        if time.time() < deadline - 30:
            def objective(params):
                ox = params[:N]
                oy = params[N:]
                return est_volume_2d_fast(ox, oy, N, num_c=20, num_grid=300)

            x0 = np.concatenate([best_x, best_y])
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            if result.fun < best_V:
                best_x = result.x[:N].copy()
                best_y = result.x[N:].copy()
                best_V = result.fun
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Build final positions
    # ------------------------------------------------------------------
    i_vals = np.arange(N)
    j_vals = np.arange(N)

    i_grid, j_grid = np.meshgrid(i_vals, j_vals, indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    bx = best_x[i_flat]
    by = best_y[j_flat]
    positions = np.stack([bx, by], axis=1)

    # Validate: shape, finite
    assert positions.shape == (N * N, 2)
    assert np.all(np.isfinite(positions))

    return positions


# EVOLVE-BLOCK-END
