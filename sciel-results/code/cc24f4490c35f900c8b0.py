"""Photonic-crystal unit-cell design: maximize TM band-4/5 gap-midgap ratio."""
import time
import numpy as np

_EPS_HIGH = 12.0
_EPS_LOW = 1.0


def _inv_eps(cell, size):
    c = np.asarray(cell, dtype=float)
    return np.where(c > 0.5, 1.0 / _EPS_HIGH, 1.0 / _EPS_LOW)


def _fft_block(inv, size, half):
    """Return (2*half+1)^2 block of 1/eps Fourier coefficients, rows/cols = Gx,Gy in [-half,half]."""
    f = np.fft.fftshift(np.fft.fft2(inv)) / (size * size)
    s = size // 2 - half
    e = size // 2 + half + 1
    return f[s:e, s:e]


def _build_C(block, Gx, Gy):
    """Coupling matrix C[i,j] = (1/eps)_{G_i - G_j} for the given plane-wave set."""
    half = max(abs(Gx.min()), abs(Gx.max()))
    dgx = Gx[:, None] - Gx[None, :]
    dgy = Gy[:, None] - Gy[None, :]
    return block[half - dgx, half - dgy]


def _band_structure(cell, size, n_pw=13, n_bands=5, k_points=None):
    """gap-midgap ratio 2*(min omega5 - max omega4)/(min omega5 + max omega4) over k_points."""
    if k_points is None:
        k_points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]
    inv = _inv_eps(cell, size)
    half = n_pw // 2
    block = _fft_block(inv, size, half)
    G = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(G, G, indexing='ij')
    Gx = Gx.flatten()
    Gy = Gy.flatten()
    N = len(Gx)
    C = _build_C(block, Gx, Gy)
    try:
        Cinv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return 0.0
    max4 = 0.0
    min5 = float('inf')
    for kx, ky in k_points:
        q2 = (Gx + kx) ** 2 + (Gy + ky) ** 2
        M = Cinv * q2[:, None]
        M = (M + M.T) * 0.5
        try:
            ev = np.linalg.eigvalsh(M)
        except np.linalg.LinAlgError:
            continue
        ev = np.sort(np.real(ev))
        if len(ev) < n_bands:
            continue
        ev = np.clip(ev, 0.0, None)
        o4 = np.sqrt(ev[n_bands - 2])
        o5 = np.sqrt(ev[n_bands - 1])
        if o4 > max4:
            max4 = o4
        if o5 < min5:
            min5 = o5
    if not np.isfinite(min5) or min5 <= 0.0:
        return 0.0
    if min5 <= max4:
        return 0.0
    return 2.0 * (min5 - max4) / (min5 + max4)


def _sym_positions(i, j, size):
    return [(i, j), (size - 1 - i, size - 1 - j), (j, i), (size - 1 - j, size - 1 - i)]


def _make_rings(size, r_list, w):
    """Concentric square rings (1 where floor(radius) in r_list)."""
    arr = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            r = int(max(abs(i - (size - 1) / 2.0), abs(j - (size - 1) / 2.0)))
            if r in r_list:
                arr[i, j] = 1
    return arr


def _make_dots(size, dots, radius):
    """Dielectric disks at given (x,y) in [0,1)."""
    arr = np.zeros((size, size))
    for x, y in dots:
        for i in range(size):
            px = (i + 0.5) / size
            for j in range(size):
                py = (j + 0.5) / size
                if (px - x) ** 2 + (py - y) ** 2 < radius ** 2:
                    arr[i, j] = 1
    return arr


def _make_rod(size, radius):
    arr = np.zeros((size, size))
    for i in range(size):
        px = (i + 0.5) / size - 0.5
        for j in range(size):
            py = (j + 0.5) / size - 0.5
            if px * px + py * py < radius ** 2:
                arr[i, j] = 1
    return arr


def _to_cell(arr, size):
    return [[1 if arr[i, j] > 0.5 else 0 for j in range(size)] for i in range(size)]


def design(problem):
    size = problem["pixels"]
    t0 = time.time()
    deadline = t0 + 195.0  # wall-clock safety margin (evaluator kills ~230s)

    def now():
        return time.time()

    k_fast = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.5)]
    k_full = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5),
              (0.25, 0.25), (0.5, 0.0), (0.0, 0.5), (0.25, 0.1)]

    def eval_fast(cell):
        return _band_structure(cell, size, n_pw=11, n_bands=5, k_points=k_fast)

    def eval_full(cell):
        return _band_structure(cell, size, n_pw=17, n_bands=5, k_points=k_full)

    # ---- candidate generators (all symmetric) ----
    candidates = []
    for r in range(2, 10):
        candidates.append(_make_rings(size, [r], 1))
        candidates.append(_make_rings(size, [r, r + 2], 1))
        candidates.append(_make_rings(size, [r, r + 1], 1))
        candidates.append(_make_rings(size, [r, r + 2, r + 4], 1))
    for r in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        candidates.append(_make_rod(size, r))
    for r in [0.10, 0.15, 0.20]:
        for dots in [
            [(0.25, 0.25)],
            [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
            [(0.5, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.5)],
            [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)],
        ]:
            candidates.append(_make_dots(size, dots, r))

    best_cell = None
    best_score = -1.0
    for arr in candidates:
        if now() > deadline:
            break
        cell = _to_cell(arr, size)
        s = eval_fast(cell)
        if s > best_score:
            best_score = s
            best_cell = cell
    if best_cell is None:
        best_cell = _to_cell(_make_rings(size, [5], 1), size)
        best_score = eval_fast(best_cell)

    # ---- random-walk local search (symmetric orbit flips) ----
    arr = np.asarray(best_cell, dtype=float)
    for step in range(400000):
        if now() > deadline - 25.0:
            break
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        pos = _sym_positions(i, j, size)
        old = [arr[p[0], p[1]] for p in pos]
        nv = 1.0 - arr[pos[0][0], pos[0][1]]
        for p in pos:
            arr[p[0], p[1]] = nv
        cell = _to_cell(arr, size)
        s = eval_fast(cell)
        if s > best_score:
            best_score = s
            best_cell = cell
        else:
            for p, ov in zip(pos, old):
                arr[p[0], p[1]] = ov

    # ---- final full-resolution refinement ----
    if best_cell is not None and now() < deadline - 5.0:
        fs = eval_full(best_cell)
        if fs > best_score:
            best_score = fs
    return best_cell
