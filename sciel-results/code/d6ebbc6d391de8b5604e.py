"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  1. Fast exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. A pool of physically-motivated layer sequences (quarter-wave stacks, symmetric
     multi-layer, graded, and "random-ish" patterns) so we explore many structures.
  3. Stage A: coarse global search over layer counts + a few thickness parameters
     (coarse grid + random restarts) to find good basins.
  4. Stage B: Nelder-Mead local polish on the best candidates (and a few others).
  5. Explicit deadline guard: return the best feasible result found so far.
"""

import time
import math
import itertools
import numpy as np

# ---------------------------------------------------------------------------
# Reflectance model (thin-film characteristic matrix, normal incidence)
# ---------------------------------------------------------------------------
def _reflectance_pct(n_amb, n_sub, n_layers, d_layers, wl):
    """Mean reflectance (percent) over wavelengths wl for the given stack.

    n_layers: list/array of refractive indices (air -> substrate order)
    d_layers: list/array of physical thicknesses (nm)
    wl:       array of wavelengths (nm)
    Returns a scalar: mean reflectance in percent over wl.
    """
    n_layers = np.asarray(n_layers, dtype=float)
    d_layers = np.asarray(d_layers, dtype=float)
    wl = np.asarray(wl, dtype=float)

    n1 = n_layers[0] if n_layers.size else n_amb
    n0 = n_amb

    # 2*pi*d / (lambda * n)
    beta = (2.0 * math.pi * d_layers) / (wl[:, None] * n_layers[None, :])
    c = np.cos(beta)
    s = np.sin(beta)

    # Layer matrices M_j = [[cos, i sin/n_j], [i n_j sin, cos]]
    # We'll multiply M = M_1 M_2 ... M_N
    m11 = np.ones_like(wl)
    m12 = np.zeros_like(wl)
    m21 = np.zeros_like(wl)
    m22 = np.ones_like(wl)

    for j in range(n_layers.size):
        n_j = n_layers[j]
        c_j = c[:, j]
        s_j = s[:, j]
        # M = [[c, i s/n], [i n s, c]]
        a = c_j
        b = 1j * s_j / n_j
        cc = 1j * n_j * s_j
        d = c_j

        # new = M_old * M
        nm11 = m11 * a + m12 * cc
        nm12 = m11 * b + m12 * d
        nm21 = m21 * a + m22 * cc
        nm22 = m21 * b + m22 * d
        m11, m12, m21, m22 = nm11, nm12, nm21, nm22

    # Boundary matrices: air side and substrate side
    # Y = (n0*m11 + n0*n1*m12) / (m11 + n1*m12)
    # r = (Y - n_sub) / (Y + n_sub)
    n0 = float(n_amb)
    n_sub = float(n_sub)

    Y = (n0 * m11 + n0 * n1 * m12) / (m11 + n1 * m12)
    r = (Y - n_sub) / (Y + n_sub)
    R = np.abs(r) ** 2
    return float(np.mean(R) * 100.0)


# ---------------------------------------------------------------------------
# Deadline guard
# ---------------------------------------------------------------------------
class _Deadline:
    def __init__(self, seconds):
        self.end = time.time() + seconds
        self.best = None
        self.best_val = float("inf")

    def expired(self, margin=0.0):
        return (time.time() - self.end) > -margin

    def record(self, materials, thicknesses, val):
        if val < self.best_val:
            self.best_val = val
            self.best = (list(materials), list(thicknesses))


# ---------------------------------------------------------------------------
# Layer sequence generators (return a list of 'L'/'H' of length N)
# ---------------------------------------------------------------------------
def _seq_qw(N):
    """Alternating starting with L (quarter-wave-like)."""
    return ['L' if (i % 2 == 0) else 'H' for i in range(N)]

def _seq_qw_H(N):
    return ['H' if (i % 2 == 0) else 'L' for i in range(N)]

def _seq_sym(N):
    """Symmetric about the middle, starting with L."""
    half = (N + 1) // 2
    first = ['L' if (i % 2 == 0) else 'H' for i in range(half)]
    if N % 2 == 0:
        return first + first[::-1]
    else:
        return first + first[-2::-1]

def _seq_sym_H(N):
    half = (N + 1) // 2
    first = ['H' if (i % 2 == 0) else 'L' for i in range(half)]
    if N % 2 == 0:
        return first + first[::-1]
    else:
        return first + first[-2::-1]

def _seq_graded(N, nL_target=0.4):
    """More L layers near the air side, more H near the substrate (graded index)."""
    seq = []
    for i in range(N):
        # fraction of "L-ness" decreases from air to substrate
        frac = 1.0 - (i / max(1, N - 1))
        # bias: L if frac > threshold
        if frac > (1.0 - nL_target):
            seq.append('L')
        else:
            seq.append('H')
    return seq

def _seq_graded_rev(N, nH_target=0.4):
    seq = []
    for i in range(N):
        frac = i / max(1, N - 1)
        if frac > (1.0 - nH_target):
            seq.append('H')
        else:
            seq.append('L')
    return seq

def _seq_random(N, rng):
    return [rng.choice(['L', 'H']) for _ in range(N)]


# ---------------------------------------------------------------------------
# Main design
# ---------------------------------------------------------------------------
def design(problem):
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mats = problem["materials"]
    nL = mats['L']
    nH = mats['H']
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    # 401 equally spaced wavelengths
    wl = np.linspace(wl_min, wl_max, 401)

    # Deadline: 60s CPU budget per contract; use a conservative 50s guard
    deadline = _Deadline(50.0)

    # Baseline: single quarter-wave L layer (the seed)
    centre = 0.5 * (wl_min + wl_max)
    base_th = centre / (4.0 * nL)
    base_mat = ['L']
    base_val = _reflectance_pct(n_amb, n_sub, [nL], [base_th], wl)
    deadline.record(base_mat, [base_th], base_val)

    # -----------------------------------------------------------------------
    # Stage A: coarse global search
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(12345)

    # Candidate sequences: a curated pool across layer counts
    candidate_seqs = []
    for N in range(1, max_layers + 1):
        for gen in (_seq_qw, _seq_qw_H, _seq_sym, _seq_sym_H, _seq_graded, _seq_graded_rev):
            try:
                seq = gen(N)
            except Exception:
                continue
            if len(seq) == N:
                candidate_seqs.append((N, seq))
        # A couple of random sequences per N for diversity
        for _ in range(2):
            candidate_seqs.append((N, _seq_random(N, rng)))

    # Remove duplicates
    seen = set()
    uniq = []
    for N, seq in candidate_seqs:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            uniq.append((N, seq))
    candidate_seqs = uniq

    # Coarse thickness grid (nm)
    coarse_grid = [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0,
                   200.0, 220.0, 240.0, 260.0, 280.0, 300.0]

    # For each candidate, do a coarse random search over thicknesses
    # We'll keep the top-K candidates by coarse best value for polishing.
    top_candidates = []  # list of (val, N, seq, thicks)

    for N, seq in candidate_seqs:
        if deadline.expired(margin=5.0):
            break
        n_vec = np.array([nL if m == 'L' else nH for m in seq])
        # Coarse random restarts: try a few random thickness vectors
        best_local = float("inf")
        best_th = None
        n_restart = 8
        for _ in range(n_restart):
            if deadline.expired(margin=5.0):
                break
            # Random thicknesses, biased toward quarter-wave values
            th = np.empty(N)
            for j in range(N):
                # quarter-wave at centre for that index
                qw = centre / (4.0 * n_vec[j])
                # random around it
                th[j] = qw * (0.7 + 0.6 * rng.random())
                th[j] = min(max(th[j], 20.0), max_layer)
            # Check total thickness constraint
            if th.sum() > max_total:
                # scale down
                th = th * (max_total / th.sum())
            val = _reflectance_pct(n_amb, n_sub, n_vec, th, wl)
            if val < best_local:
                best_local = val
                best_th = th.copy()
        if best_th is not None and best_local < deadline.best_val:
            deadline.record(seq, list(best_th), best_local)
            top_candidates.append((best_local, N, seq, best_th))

    # Sort candidates by best coarse value
    top_candidates.sort(key=lambda x: x[0])

    # -----------------------------------------------------------------------
    # Stage B: Nelder-Mead polish on the best candidates
    # -----------------------------------------------------------------------
    def _nm_polish(N, seq, x0, max_iter=200):
        """Nelder-Mead on thicknesses (nm), with constraint projection."""
        n_vec = np.array([nL if m == 'L' else nH for m in seq])
        x = np.array(x0, dtype=float)

        def obj(xv):
            # Project into feasible region
            xv = np.clip(xv, 0.0, max_layer)
            if xv.sum() > max_total:
                xv = xv * (max_total / xv.sum())
            return _reflectance_pct(n_amb, n_sub, n_vec, xv, wl)

        # Simplex
        simplex = [x.copy()]
        for i in range(N):
            xi = x.copy()
            step = max(5.0, 0.1 * x[i])
            xi[i] += step
            simplex.append(xi)

        fvals = [obj(v) for v in simplex]
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        best_x = simplex[0].copy()
        best_f = fvals[0]

        for it in range(max_iter):
            if deadline.expired(margin=3.0):
                break
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = fvals[-1]

            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - worst)
            fr = obj(xr)

            if fr < fvals[0]:
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (centroid - worst)
                fe = obj(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                rho = 0.5
                xc = centroid + rho * (worst - centroid)
                fc = obj(xc)
                if fc < f_worst:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, N + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = obj(simplex[i])

            # Reorder
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Convergence check
            if max(fvals) - min(fvals) < 1e-6:
                break

        # Final projection
        best_x = np.clip(best_x, 0.0, max_layer)
        if best_x.sum() > max_total:
            best_x = best_x * (max_total / best_x.sum())
        return best_x, obj(best_x)

    # Polish the top candidates
    n_polish = min(6, len(top_candidates))
    for idx in range(n_polish):
        if deadline.expired(margin=3.0):
            break
        val, N, seq, th = top_candidates[idx]
        try:
            new_th, new_val = _nm_polish(N, seq, th, max_iter=150)
        except Exception:
            continue
        if new_val < deadline.best_val:
            deadline.record(seq, list(new_th), new_val)

    # -----------------------------------------------------------------------
    # Stage C: a few extra random full searches if time remains
    # -----------------------------------------------------------------------
    if not deadline.expired(margin=5.0):
        for _ in range(20):
            if deadline.expired(margin=3.0):
                break
            N = int(rng.integers(2, max_layers + 1))
            seq = _seq_random(N, rng)
            n_vec = np.array([nL if m == 'L' else nH for m in seq])
            th = np.empty(N)
            for j in range(N):
                qw = centre / (4.0 * n_vec[j])
                th[j] = qw * (0.7 + 0.6 * rng.random())
                th[j] = min(max(th[j], 20.0), max_layer)
            if th.sum() > max_total:
                th = th * (max_total / th.sum())
            val = _reflectance_pct(n_amb, n_sub, n_vec, th, wl)
            if val < deadline.best_val:
                deadline.record(seq, list(th), val)
                # Quick polish
                if not deadline.expired(margin=3.0):
                    try:
                        new_th, new_val = _nm_polish(N, seq, th, max_iter=100)
                        if new_val < deadline.best_val:
                            deadline.record(seq, list(new_th), new_val)
                    except Exception:
                        pass

    # -----------------------------------------------------------------------
    # Return best feasible result
    # -----------------------------------------------------------------------
    if deadline.best is None:
        return base_mat, [base_th]

    materials, thicknesses = deadline.best
    # Final feasibility check
    thicknesses = [min(max(float(t), 0.0), max_layer) for t in thicknesses]
    if sum(thicknesses) > max_total:
        scale = max_total / sum(thicknesses)
        thicknesses = [t * scale for t in thicknesses]
    if len(materials) > max_layers:
        materials = materials[:max_layers]
        thicknesses = thicknesses[:max_layers]
    return materials, thicknesses
