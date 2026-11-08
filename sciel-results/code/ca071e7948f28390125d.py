"""Bootstrap DIRECT solution: derive the Hermite-Gaussian coefficients from the
Fourier-invariant C4 uncertainty-product structure, with a hard wall-clock guard."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import eval_hermite

# Wall-clock deadline guard (evaluator kills at 210 s; keep a safety margin).
T_START = time.time()
DEADLINE = 210.0 - 15.0  # 195 s hard stop


def _now():
    return time.time()


def _time_left():
    return DEADLINE - _now()


# Physically normalised probabilists' Hermite polynomials.
# He_n(x) is the probabilists' Hermite polynomial (recurrence He_{n+1}=x He_n - n He_{n-1}).
# The Fourier-invariant (self-dual up to sign) Gaussian-Hermite eigenfunctions use the
# physicists' form; here we work with a fixed basis so only the relative weights matter
# for the uncertainty product (a scale-invariant ratio). We use the standard physicists'
# H_n with the 2^{n/2} sqrt(n! pi) normalisation factor folded into the basis.
def _H(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    return eval_hermite(n, x)


def _norm_factor(n):
    """2^{n/2} * sqrt(n! * pi) : L2 norm of H_n(x) exp(-x^2/2) over R."""
    return (2.0 ** (0.5 * n)) * math.sqrt(math.factorial(n) * math.pi)


def _make_basis():
    """Return basis vectors and pre-computed moment matrices for fast variance computation.
    
    Basis: φ_n(x) = H_n(x) * exp(-x²/2) / ||H_n(x) exp(-x²/2)||_2
    Pre-computed:
      G1[j,k] = ∫ x * φ_j(x) * φ_k(x) dx
      G2[j,k] = ∫ x² * φ_j(x) * φ_k(x) dx
    For coefficients c = [c0, c4, c8, c12], f = Σ c_j φ_j:
      E[x] = c^T G1 c
      E[x²] = c^T G2 c
      Var_x = E[x²] - (E[x])²
    """
    xs = np.linspace(-40.0, 40.0, 40001)
    dx = xs[1] - xs[0]
    w = np.exp(-0.5 * xs * xs)
    degrees = (0, 4, 8, 12)
    basis = {}
    phi = []  # list of basis function vectors
    for n in degrees:
        h = _H(n, xs)
        f = h * w
        f = f / np.sqrt(np.sum(f * f) * dx + 1e-300)
        basis[n] = f
        phi.append(f)
    
    # Pre-compute moment matrices G1 and G2 (4x4)
    N = len(degrees)
    G1 = np.zeros((N, N))
    G2 = np.zeros((N, N))
    for j in range(N):
        for k in range(j, N):
            prod = phi[j] * phi[k]
            g1 = np.sum(xs * prod) * dx
            g2 = np.sum(xs * xs * prod) * dx
            G1[j, k] = g1
            G1[k, j] = g1
            G2[j, k] = g2
            G2[k, j] = g2
    
    return xs, dx, w, basis, G1, G2


XS, DX, W, BASIS, G1, G2 = _make_basis()


def _coeff12(c0, c4, c8):
    """Derive the H12 weight so that the combination has P(0)=0 (g(0)=0).
    g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    => c12 = -(c0*H0(0) + c4*H4(0) + c8*H8(0)) / H12(0).
    We work in the unnormalised polynomial values at 0."""
    h0 = _H(0, 0.0)
    h4 = _H(4, 0.0)
    h8 = _H(8, 0.0)
    h12 = _H(12, 0.0)
    if abs(h12) < 1e-15:
        return 0.0
    return -(c0 * h0 + c4 * h4 + c8 * h8) / h12


def _uncertainty_product(c0, c4, c8):
    """Compute the C4 upper bound: A(f)*A(Fourier(f)) for the constructed f.

    Using pre-computed moment matrices for O(1) evaluation.
    The Fourier transform of each Hermite-Gaussian eigenfunction is ±itself,
    so Var_p = Var_x, and the product is Var_x."""
    c12 = _coeff12(c0, c4, c8)
    # Coefficient vector in order [0, 4, 8, 12]
    c = np.array([c0, c4, c8, c12])
    
    # E[x] = c^T G1 c
    ex = float(c @ G1 @ c)
    # E[x²] = c^T G2 c
    ex2 = float(c @ G2 @ c)
    var_x = ex2 - ex * ex
    var_x = max(var_x, 1e-15)
    return var_x


def _score(c0, c4, c8):
    """The upper bound C4 value (lower is better)."""
    try:
        return _uncertainty_product(c0, c4, c8)
    except Exception:
        return float('inf')


def _score_vec(c0, c4, c8):
    """Vectorized scoring for multiple coefficient sets at once.
    Returns array of uncertainty products."""
    c12_arr = _coeff12_vec(c0, c4, c8)
    # c0, c4, c8, c12 can be scalars or arrays
    c0 = np.asarray(c0, dtype=float)
    c4 = np.asarray(c4, dtype=float)
    c8 = np.asarray(c8, dtype=float)
    c12 = np.asarray(c12_arr, dtype=float)
    
    # E[x] = c0²*G1[0,0] + 2*c0*c4*G1[0,1] + 2*c0*c8*G1[0,2] + 2*c0*c12*G1[0,3]
    #       + c4²*G1[1,1] + 2*c4*c8*G1[1,2] + 2*c4*c12*G1[1,3]
    #       + c8²*G1[2,2] + 2*c8*c12*G1[2,3] + c12²*G1[3,3]
    ex = (c0*c0*G1[0,0] + 2*c0*c4*G1[0,1] + 2*c0*c8*G1[0,2] + 2*c0*c12*G1[0,3]
          + c4*c4*G1[1,1] + 2*c4*c8*G1[1,2] + 2*c4*c12*G1[1,3]
          + c8*c8*G1[2,2] + 2*c8*c12*G1[2,3] + c12*c12*G1[3,3])
    ex2 = (c0*c0*G2[0,0] + 2*c0*c4*G2[0,1] + 2*c0*c8*G2[0,2] + 2*c0*c12*G2[0,3]
           + c4*c4*G2[1,1] + 2*c4*c8*G2[1,2] + 2*c4*c12*G2[1,3]
           + c8*c8*G2[2,2] + 2*c8*c12*G2[2,3] + c12*c12*G2[3,3])
    var_x = ex2 - ex * ex
    return np.maximum(var_x, 1e-15)


def _coeff12_vec(c0, c4, c8):
    """Vectorized version of _coeff12."""
    h0 = _H(0, 0.0)
    h4 = _H(4, 0.0)
    h8 = _H(8, 0.0)
    h12 = _H(12, 0.0)
    if abs(h12) < 1e-15:
        return np.zeros_like(np.asarray(c0, dtype=float))
    return -(np.asarray(c0)*h0 + np.asarray(c4)*h4 + np.asarray(c8)*h8) / h12


def _coarse_grid():
    """Direct structured sweep over the three free coefficients, vectorized."""
    best = None
    best_val = float('inf')
    # Fix c8 = 1 and search c0/c8, c4/c8 ratios.
    ratios_c0 = np.linspace(-30.0, 30.0, 121)
    ratios_c4 = np.linspace(-30.0, 30.0, 121)
    
    # Vectorize: evaluate all combinations at once
    r0_grid, r4_grid = np.meshgrid(ratios_c0, ratios_c4, indexing='ij')
    r0_flat = r0_grid.ravel()
    r4_flat = r4_grid.ravel()
    
    # Process in chunks to avoid memory issues
    chunk_size = 10000
    for start in range(0, len(r0_flat), chunk_size):
        if _time_left() < 15.0:
            break
        end = min(start + chunk_size, len(r0_flat))
        vals = _score_vec(r0_flat[start:end], r4_flat[start:end], np.ones(end - start))
        local_min_idx = np.argmin(vals)
        local_min_val = vals[local_min_idx]
        if local_min_val < best_val:
            best_val = local_min_val
            idx = start + local_min_idx
            best = (r0_flat[idx], r4_flat[idx], 1.0)
    return best, best_val


def _local_refine(c0, c4, c8, iters=400):
    """Deterministic coordinate + Nelder-Mead style refinement around a start point."""
    best = (c0, c4, c8)
    best_val = _score(c0, c4, c8)
    # Simple Nelder-Mead with a few iterations, bounded.
    pts = [
        (c0, c4, c8),
        (c0 + 0.5, c4, c8),
        (c0 - 0.5, c4, c8),
        (c0, c4 + 0.5, c8),
        (c0, c4 - 0.5, c8),
        (c0, c4, c8 + 0.5),
        (c0, c4, c8 - 0.5),
    ]
    vals = [_score(*p) for p in pts]
    for _ in range(iters):
        if _time_left() < 15.0:
            break
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        centroid = tuple(np.mean(pts[:-1], axis=0))
        alpha = 1.0
        reflection = tuple(centroid[i] + alpha * (centroid[i] - pts[-1][i]) for i in range(3))
        vr = _score(*reflection)
        if vr < vals[0]:
            # expansion
            gamma = 2.0
            expansion = tuple(centroid[i] + gamma * (reflection[i] - pts[-1][i]) for i in range(3))
            ve = _score(*expansion)
            if ve < vr:
                pts[-1], vals[-1] = expansion, ve
            else:
                pts[-1], vals[-1] = reflection, vr
        elif vr < vals[-2]:
            pts[-1], vals[-1] = reflection, vr
        else:
            # contraction
            rho = 0.5
            contraction = tuple(centroid[i] + rho * (pts[-1][i] - centroid[i]) for i in range(3))
            vc = _score(*contraction)
            if vc < vals[-1]:
                pts[-1], vals[-1] = contraction, vc
            else:
                # shrink toward best
                for i in range(1, 3):
                    pts[i] = tuple(pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(3))
                    vals[i] = _score(*pts[i])
        cur_best = min(range(3), key=lambda i: vals[i])
        if vals[cur_best] < best_val:
            best_val = vals[cur_best]
            best = pts[cur_best]
    return best, best_val


def construct():
    from scipy.optimize import minimize

    # Seed prior-art coefficients as a fallback / initial guess.
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])

    best = tuple(seed.tolist())
    best_val = _score(*best)

    # Phase 1: coarse structured sweep (2D via scale-invariance, c8 fixed to 1).
    if _time_left() > 15.0:
        grid_best, grid_val = _coarse_grid()
        if grid_best is not None and grid_val < best_val:
            best, best_val = grid_best, grid_val
        # Also try the seed-scaled point and a few hand-derived structured points.
        for cand in [
            (-113.0, 1.0, 3240.0 / 100.0),
            (-20.0, 4.0, 1.0),
            (-50.0, 10.0, 1.0),
            (0.0, 0.0, 1.0),
            (-1.13, 0.04, 1.0),
            (-3.0, 0.5, 1.0),
            (-5.0, 1.0, 1.0),
            (-10.0, 2.0, 1.0),
        ]:
            v = _score(*cand)
            if v < best_val:
                best, best_val = cand, v

    # Phase 2: Use scipy.optimize.minimize for thorough local search from multiple starts.
    def _obj_2d(params):
        """Objective in 2D: params = [c0/c8, c4/c8], c8=1."""
        return _score(params[0], params[1], 1.0)

    # Multiple starting points for the 2D optimization, including neighborhood of best
    best_ratio0 = best[0] / best[2] if abs(best[2]) > 1e-15 else 0.0
    best_ratio4 = best[1] / best[2] if abs(best[2]) > 1e-15 else 0.0
    starts = [
        np.array([best_ratio0, best_ratio4]),
        np.array([best_ratio0 + 0.5, best_ratio4]),
        np.array([best_ratio0 - 0.5, best_ratio4]),
        np.array([best_ratio0, best_ratio4 + 0.5]),
        np.array([best_ratio0, best_ratio4 - 0.5]),
        np.array([-1.13, 0.04]),  # seed ratios
        np.array([-2.0, 0.5]),
        np.array([-5.0, 1.0]),
        np.array([-10.0, 2.0]),
        np.array([0.0, 0.0]),
        np.array([-3.0, -0.5]),
        np.array([3.0, 0.5]),
    ]

    for start in starts:
        if _time_left() < 8.0:
            break
        try:
            res = minimize(_obj_2d, start, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best = (res.x[0], res.x[1], 1.0)
                best_val = res.fun
        except Exception:
            pass

    # Phase 3: Final high-precision polish using BFGS with gradient approximation
    if _time_left() > 10.0:
        def _obj_2d_polish(params):
            return _score(params[0], params[1], 1.0)
        
        try:
            res = minimize(_obj_2d_polish, np.array([best[0]/best[2] if abs(best[2])>1e-15 else 0.0,
                                                      best[1]/best[2] if abs(best[2])>1e-15 else 0.0]),
                          method='BFGS',
                          options={'gtol': 1e-14, 'maxiter': 1000})
            if res.fun < best_val:
                best = (res.x[0], res.x[1], 1.0)
                best_val = res.fun
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead refinement
    if _time_left() > 8.0:
        ref_best, ref_val = _local_refine(*best, iters=1000)
        if ref_val < best_val:
            best, best_val = ref_best, ref_val

    # Return the three free coefficients (degrees 0, 4, 8).
    return np.array([best[0], best[1], best[2]])
# EVOLVE-BLOCK-END
