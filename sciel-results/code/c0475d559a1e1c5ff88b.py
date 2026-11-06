import time
import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize

_DEADLINE = 0.0
_SIZE = 32
_EPS_HIGH = 12.0

# ---------------------------------------------------------------------------
# Fast TM band-gap surrogate (plane-wave Galerkin, TM)
# ---------------------------------------------------------------------------
def _build_pw(pw_radius):
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx * Gx + Gy * Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    size_d = 4 * pw_radius + 1
    offset_d = 2 * pw_radius
    return G_arr, Gx_diff, Gy_diff, size_d, offset_d, n_pw


def _compute_gap_fast(cell, n_k=10, pw_radius=11):
    N = _SIZE
    inv_eps = 1.0 / (1.0 + (EPS_HIGH - 1.0) * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)

    G_arr, Gx_diff, Gy_diff, size_d, offset_d, n_pw = _build_pw(pw_radius)

    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]

    k_points = [
        (0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
        (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5),
        (0.15, 0.15), (0.25, 0.25), (0.35, 0.35), (0.4, 0.4),
    ][:n_k]

    max_w4 = 0.0
    min_w5 = float('inf')

    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T

        inv_eps_mat = inv_eps_full[
            Gx_diff.astype(int) + offset_d,
            Gy_diff.astype(int) + offset_d
        ]

        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2

        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except Exception:
            pass

    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)


# ---------------------------------------------------------------------------
# Symmetric parametric shape families (all respect the full square group)
# ---------------------------------------------------------------------------
def _ring_cell(r_in, r_out, thick=0.0):
    """Annular dielectric ring: dielectric where r_in < r <= r_out (plus core if r_in<0)."""
    N = _SIZE
    cell = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            r = np.sqrt(x * x + y * y)
            if r_in < 0:
                cell[i, j] = 1 if r <= r_out else 0
            else:
                cell[i, j] = 1 if (r_in < r <= r_out) else 0
    return cell


def _sector_cell(n_sectors, fill_frac, r_max):
    """n_sectors radial sectors (square-symmetric count) filled up to r_max.
    fill_frac controls angular width within each sector slot."""
    N = _SIZE
    cell = np.zeros((N, N), dtype=int)
    # For square symmetry, n_sectors must be multiple of 4
    n_sectors = max(4, 4 * int(round(n_sectors / 4.0)))
    slot = 2 * np.pi / n_sectors
    half_fill = (fill_frac * slot) / 2.0
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            r = np.sqrt(x * x + y * y)
            if r > r_max:
                continue
            ang = np.arctan2(y, x)
            # fold into [0, slot)
            a = ang % slot
            if a < half_fill or a > slot - half_fill:
                cell[i, j] = 1
    return cell


def _grid_cell(n_rings, ring_width, core_r):
    """Concentric rings with alternating fill; core_r controls inner solid disc."""
    N = _SIZE
    cell = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            r = np.sqrt(x * x + y * y)
            if r <= core_r:
                cell[i, j] = 1
                continue
            # ring index
            ring_idx = int((r - core_r) / ring_width) if ring_width > 0 else 0
            if ring_idx < n_rings and (ring_idx % 2 == 0):
                cell[i, j] = 1
    return cell


def _make_symmetric(cell):
    """Average over all 8 symmetry operations and threshold."""
    c = np.array(cell, dtype=float)
    c_t = c.T
    c_r = c[::-1, :]
    c_c = c[:, ::-1]
    c_rt = c[::-1, ::-1]
    c_tr = c.T[::-1, :]
    c_tc = c.T[:, ::-1]
    c_rtc = c[::-1, ::-1].T
    c_averaged = (c + c_t + c_r + c_c + c_rt + c_tr + c_tc + c_rtc) / 8.0
    c_sym = (c_averaged > 0.5).astype(int)
    return c_sym


# ---------------------------------------------------------------------------
# Parametric candidate generators -> (cell, params)
# ---------------------------------------------------------------------------
def _gen_from_family(family, params):
    """Return a symmetric 32x32 int cell for the given family and params."""
    if family == 0:
        # concentric ring: params = [r_in, r_out]
        r_in = float(np.clip(params[0], -0.01, 0.45))
        r_out = float(np.clip(params[1], 0.05, 0.48))
        if r_out <= r_in:
            r_out = r_in + 0.02
        cell = _ring_cell(r_in, r_out)
    elif family == 1:
        # radial sectors: params = [n_sectors, fill_frac, r_max]
        n_sec = int(np.clip(round(params[0]), 4, 32))
        fill = float(np.clip(params[1], 0.05, 0.95))
        r_max = float(np.clip(params[2], 0.05, 0.48))
        cell = _sector_cell(n_sec, fill, r_max)
    elif family == 2:
        # grid rings: params = [n_rings, ring_width, core_r]
        n_r = int(np.clip(round(params[0]), 1, 5))
        rw = float(np.clip(params[1], 0.03, 0.15))
        core = float(np.clip(params[2], 0.0, 0.3))
        cell = _grid_cell(n_r, rw, core)
    else:
        cell = _ring_cell(-0.01, 0.2)

    cell = _make_symmetric(cell)
    return cell


# ---------------------------------------------------------------------------
# Nelder-Mead optimizer over continuous params for a fixed family
# ---------------------------------------------------------------------------
def _optimize_family(family, x0, deadline, n_k=8, pw_radius=10):
    def objective(params):
        if time.time() > deadline - 30:
            return 1e6
        try:
            cell = _gen_from_family(family, params)
            return -_compute_gap_fast(cell, n_k=n_k, pw_radius=pw_radius)
        except Exception:
            return 1e6

    bounds = {
        0: [(-0.01, 0.45), (0.05, 0.48)],
        1: [(4, 32), (0.05, 0.95), (0.05, 0.48)],
        2: [(1, 5), (0.03, 0.15), (0.0, 0.3)],
    }[family]

    n_params = len(x0)
    res = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': 60, 'xatol': 0.01, 'fatol': 0.005,
                 'adaptive': True}
    )
    best_params = res.x
    best_cell = _gen_from_family(family, best_params)
    best_score = -res.fun if res.fun < 1e5 else -1.0
    return best_cell, best_score


# ---------------------------------------------------------------------------
# Main design entry point
# ---------------------------------------------------------------------------
def design(problem):
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 195.0  # safety margin under 230s wall-clock

    size = problem["pixels"]
    assert size == 32

    # Initial diverse candidates across families
    initial_candidates = [
        (0, [-0.01, 0.15]),   # small rod
        (0, [-0.01, 0.20]),   # medium rod
        (0, [-0.01, 0.25]),   # larger rod
        (0, [0.10, 0.25]),    # ring
        (0, [0.15, 0.30]),    # wider ring
        (1, [8, 0.5, 0.30]),  # 8 sectors
        (1, [12, 0.4, 0.35]), # 12 sectors
        (1, [16, 0.3, 0.40]), # 16 sectors
        (2, [2, 0.08, 0.10]), # 2 rings + core
        (2, [3, 0.06, 0.05]), # 3 rings + core
    ]

    best_cell = None
    best_score = -1.0

    # Quick screen
    for fam, params in initial_candidates:
        if time.time() > _DEADLINE - 120:
            break
        try:
            cell = _gen_from_family(fam, params)
            score = _compute_gap_fast(cell, n_k=8, pw_radius=10)
            if score > best_score:
                best_score = score
                best_cell = cell
        except Exception:
            pass

    if best_cell is None:
        best_cell = _make_symmetric(_ring_cell(-0.01, 0.20))
        best_score = _compute_gap_fast(best_cell, n_k=8, pw_radius=10)

    # Nelder-Mead optimization from the best starting family, plus a few others
    # Determine which family the best candidate came from
    best_family = 0
    best_params = [-0.01, 0.20]
    for fam, params in initial_candidates:
        if time.time() > _DEADLINE - 120:
            break
        try:
            cell = _gen_from_family(fam, params)
            score = _compute_gap_fast(cell, n_k=8, pw_radius=10)
            if abs(score - best_score) < 1e-9:
                best_family = fam
                best_params = list(params)
                break
        except Exception:
            pass

    # Optimize the best family
    if time.time() < _DEADLINE - 80:
        try:
            opt_cell, opt_score = _optimize_family(
                best_family, best_params, _DEADLINE, n_k=8, pw_radius=10
            )
            if opt_score > best_score:
                best_score = opt_score
                best_cell = opt_cell
        except Exception:
            pass

    # Optimize a second family (the other top one)
    if time.time() < _DEADLINE - 50:
        second_family = 1 if best_family != 1 else 2
        second_params = [8, 0.5, 0.30] if second_family == 1 else [2, 0.08, 0.10]
        try:
            opt_cell2, opt_score2 = _optimize_family(
                second_family, second_params, _DEADLINE, n_k=8, pw_radius=10
            )
            if opt_score2 > best_score:
                best_score = opt_score2
                best_cell = opt_cell2
        except Exception:
            pass

    # Third family if time permits
    if time.time() < _DEADLINE - 35:
        third_family = 2 if best_family != 2 else 1
        third_params = [3, 0.06, 0.05] if third_family == 2 else [16, 0.3, 0.40]
        try:
            opt_cell3, opt_score3 = _optimize_family(
                third_family, third_params, _DEADLINE, n_k=8, pw_radius=10
            )
            if opt_score3 > best_score:
                best_score = opt_score3
                best_cell = opt_cell3
        except Exception:
            pass

    # Final high-fidelity verification
    if best_cell is not None and time.time() < _DEADLINE - 10:
        final_score = _compute_gap_fast(best_cell, n_k=12, pw_radius=12)

    # Ensure symmetry and return as nested list
    best_cell = _make_symmetric(best_cell)
    return best_cell.tolist()
