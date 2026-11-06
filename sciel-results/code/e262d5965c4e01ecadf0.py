"""Dzugutov cluster global minima: icosahedral-shell seeds + basin-hopping with BFGS relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 290.0  # safety margin under the 330 s kill (task says 300s, but evaluator kills at 330s)

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16
_MIN_SEP = 0.5


def _remaining():
    return TIME_LIMIT - (time.time() - _START)


def _energy(pos):
    """Vectorised Dzugutov energy (flat array, n*3)."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    diff = p[:, None, :] - p[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    ri = r[iu]
    e = np.zeros(ri.shape, dtype=float)
    m1 = ri < _a
    m2 = (ri >= _a) & (ri < _b)
    if np.any(m1):
        r1 = ri[m1]
        e[m1] = _A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a))
    if np.any(m2):
        r2 = ri[m2]
        e[m2] = _B * np.exp(_d / (r2 - _b))
    return float(e.sum())


def _grad_fn(x):
    """Gradient via the provided helper, reshaped to flat."""
    p = np.asarray(x, dtype=float).reshape(-1, 3)
    try:
        _, g = dzugutov_energy_gradient(p)
        g = np.asarray(g, dtype=float).reshape(-1)
        if np.all(np.isfinite(g)):
            return g
    except Exception:
        pass
    # finite-difference fallback
    n = p.size
    h = 1e-5
    e0 = _energy(x)
    g = np.empty(n, dtype=float)
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (_energy(xp) - _energy(xm)) / (2 * h)
    return g


def _min_sep_ok(pos):
    """Check if all pairwise distances are >= _MIN_SEP."""
    if pos.shape[0] < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(pos.shape[0], k=1)
    return r[iu].min() >= _MIN_SEP - 1e-9


def _make_icosahedron(scale):
    """12 vertices of a regular icosahedron scaled so nearest-neighbour distance ~ scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    # edge length of this icosahedron is 2
    v = v * (scale / 2.0)
    return v


def _icosahedral_seeds(n, base=1.10):
    """Generate several icosahedral-shell starting configurations for cluster size n."""
    seeds = []
    # Core (1) + first shell (12)
    r1 = base
    r2 = base * 1.7320508  # ~sqrt(3)
    r3 = base * 2.0
    
    dirs = _make_icosahedron(1.0)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / (norms + 1e-12)

    def build(radii):
        pts = [np.zeros(3)]
        # shell 1: 12
        for d in dirs:
            pts.append(d * r1)
        placed = 13
        shell_id = 1
        while placed < n:
            shell_id += 1
            r = radii[min(shell_id, len(radii)) - 1]
            # distribute on a sphere: use icosahedral dirs plus radial jitter / higher shell
            for d in dirs:
                if placed >= n:
                    break
                # offset direction slightly per shell to avoid coincident sites
                off = 0.12 * (shell_id - 1)
                dd = d + off * np.array([0.0, 0.0, 1.0])
                dd = dd / (np.linalg.norm(dd) + 1e-12)
                pts.append(dd * r)
                placed += 1
            if shell_id > 8:
                break
        return np.array(pts[:n])

    # Try different second-shell radii
    for r2f in (1.60, 1.70, 1.7320508, 1.85, 2.0):
        radii = [base, base * r2f, base * 2.0]
        try:
            cfg = build(radii)
            if cfg.shape[0] == n and _min_sep_ok(cfg):
                seeds.append(cfg)
        except Exception:
            pass
    
    # Also try a few different base scales
    for b in (1.05, 1.15, 1.25):
        radii = [b, b * 1.7320508, b * 2.0]
        try:
            cfg = build(radii)
            if cfg.shape[0] == n and _min_sep_ok(cfg):
                seeds.append(cfg)
        except Exception:
            pass

    # Random-shell fallback
    rng = np.random.default_rng(n)
    for _ in range(3):
        pts = [np.zeros(3)]
        for i in range(n - 1):
            d = rng.normal(size=3)
            d = d / (np.linalg.norm(d) + 1e-12)
            r = base * (1.0 + 0.5 * (i / max(1, n - 1)))
            pts.append(d * r)
        cfg = np.array(pts)
        if _min_sep_ok(cfg):
            seeds.append(cfg)
    
    return seeds


def _relax_bfgs(pos):
    """Gradient-based BFGS relaxation; returns (energy, positions)."""
    x0 = pos.reshape(-1)
    try:
        res = minimize(
            _energy, x0, jac=_grad_fn,
            method="BFGS",
            options={"maxiter": 200, "gtol": 1e-5, "disp": False},
        )
        e = _energy(res.x)
        p = res.x.reshape(-1, 3)
        if _min_sep_ok(p):
            return e, p
    except Exception:
        pass
    return _energy(x0), pos


def _basin_hopping(pos, rng, max_iter=30, T_start=0.8, T_end=0.05, step_scale=0.25):
    """Basin hopping with BFGS relaxation. Returns best (energy, positions)."""
    best_e, best_p = _relax_bfgs(pos)
    T = T_start
    cooling = (T_start - T_end) / max(1, max_iter)
    rejected_count = 0
    
    for it in range(max_iter):
        if _remaining() < 15.0:
            break
        
        # Occasional large jump every 20 iterations
        if it % 20 == 19:
            scale = step_scale * 2.0
        else:
            scale = step_scale
        
        perturbed = best_p + rng.normal(0.0, scale, size=best_p.shape)
        
        # Ensure minimum separation after perturbation
        if not _min_sep_ok(perturbed):
            # Try to fix by nudging close pairs apart
            for _ in range(50):
                diff = perturbed[:, None, :] - perturbed[None, :, :]
                r = np.linalg.norm(diff, axis=2)
                iu = np.triu_indices(perturbed.shape[0], k=1)
                close = r[iu] < _MIN_SEP
                if not np.any(close):
                    break
                idx_i, idx_j = iu[0][close], iu[1][close]
                for i, j in zip(idx_i, idx_j):
                    d = perturbed[i] - perturbed[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-8:
                        d = rng.normal(size=3)
                    d = d / (np.linalg.norm(d) + 1e-12)
                    nudge = 0.1 * d
                    perturbed[i] += nudge
                    perturbed[j] -= nudge
                if _min_sep_ok(perturbed):
                    break
        
        if not _min_sep_ok(perturbed):
            continue
        
        e_new, p_new = _relax_bfgs(perturbed)
        
        delta_e = e_new - best_e
        accept = (delta_e < 0) or (rng.random() < np.exp(-delta_e / max(T, 1e-10)))
        
        if accept:
            best_e, best_p = e_new, p_new
            rejected_count = 0
        else:
            rejected_count += 1
            # Reset after too many rejections
            if rejected_count > 10:
                rejected_count = 0
                best_e, best_p = _relax_bfgs(best_p + rng.normal(0.0, step_scale * 1.5, size=best_p.shape))
        
        T -= cooling
        T = max(T, T_end)
    
    return best_e, best_p


def construct(instance):
    """Find low-energy Dzugutov cluster using icosahedral seeds + basin hopping."""
    size = instance["n"]
    potential = instance.get("potential", "dzugutov")
    rng = np.random.default_rng(size)
    
    # Generate diverse icosahedral seeds
    seeds = _icosahedral_seeds(size)
    if not seeds:
        # Last-resort compact random
        rad = 1.1 * size ** (1.0 / 3.0)
        pts = []
        while len(pts) < size:
            p = rng.uniform(-rad, rad, 3)
            if np.linalg.norm(p) <= rad and all(np.linalg.norm(p - q) >= 0.8 for q in pts):
                pts.append(p)
        seeds = [np.array(pts)]
    
    best_e = np.inf
    best_pos = seeds[0]
    
    # Phase 1: Relax each seed with BFGS, keep the best
    for s in seeds:
        if _remaining() < 20.0:
            break
        try:
            e, p = _relax_bfgs(s)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass
    
    # Phase 2: Basin hopping from the best structure
    if _remaining() > 30.0:
        try:
            e_bh, p_bh = _basin_hopping(best_pos, rng, max_iter=40, T_start=0.8, T_end=0.05, step_scale=0.25)
            if e_bh < best_e and _min_sep_ok(p_bh):
                best_e, best_pos = e_bh, p_bh
        except Exception:
            pass
    
    # Phase 3: If time remains, try basin hopping from a different seed
    if _remaining() > 30.0 and len(seeds) > 1:
        for s in seeds[1:4]:  # Try up to 3 more seeds
            if _remaining() < 20.0:
                break
            try:
                e, p = _relax_bfgs(s)
                if e < best_e:
                    best_e, best_pos = e, p
                if _remaining() > 25.0:
                    e_bh, p_bh = _basin_hopping(p, rng, max_iter=20, T_start=0.6, T_end=0.05, step_scale=0.25)
                    if e_bh < best_e and _min_sep_ok(p_bh):
                        best_e, best_pos = e_bh, p_bh
            except Exception:
                pass
    
    # Final safety: ensure we have a valid result
    if best_pos is None or not _min_sep_ok(best_pos):
        best_pos = seeds[0]
    
    return np.asarray(best_pos, dtype=float)
# EVOLVE-BLOCK-END
