"""Self-contained optimizer for the 12-stage Heisenberg-chain product formula.

Builds the 6-site open XXZ chain Hamiltonian (anisotropy 0.8, field 0.3),
splits it into odd-bond (A) and even-bond (B) halves each with half the field,
and optimizes a symmetric 12-stage product-formula coefficient set to minimize
the mean spectral-norm error against the exact propagator at t in {0.25,0.5,1.0}.
"""
import time
import math
import numpy as np

# ---- task-physical constants (fixed by the contract) ----
N_SITE = 6
DELTA = 0.8
H_FIELD = 0.3
T_TASK = (0.25, 0.5, 1.0)


def _build_hamiltonians():
    """Return (A, B) half-bond Hamiltonian matrices (complex, Hermitian)."""
    dim = 2 ** N_SITE
    # Pauli matrices acting on site index i (bit i, little-endian).
    # We build S^x[i], S^y[i], S^z[i] as 4x4 on a pair of sites is unnecessary;
    # instead use Kronecker products over 6 sites.
    I = np.eye(2, dtype=complex)
    sx = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=complex)
    sy = np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=complex)
    sz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)

    def kron_all(mats):
        out = np.array([[1.0 + 0.0j]])
        for m in mats:
            out = np.kron(out, m)
        return out

    Sx = [None] * N_SITE
    Sy = [None] * N_SITE
    Sz = [None] * N_SITE
    for i in range(N_SITE):
        mats = [I] * N_SITE
        mats[i] = sx
        Sx[i] = kron_all(mats)
    for i in range(N_SITE):
        mats = [I] * N_SITE
        mats[i] = sy
        Sy[i] = kron_all(mats)
    for i in range(N_SITE):
        mats = [I] * N_SITE
        mats[i] = sz
        Sz[i] = kron_all(mats)

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    # Nearest-neighbor bonds i-(i+1) for i = 0..N-2.
    for i in range(N_SITE - 1):
        J = Sx[i] @ Sx[i + 1] + Sy[i] @ Sy[i + 1] + DELTA * (Sz[i] @ Sz[i + 1])
        if i % 2 == 0:
            B += J          # even bond
        else:
            A += J          # odd bond

    # Uniform field split: half to each half, S^z on every site.
    field_half = 0.0
    for i in range(N_SITE):
        # field term: (H/2) * 2*S^z = H * S^z? Contract: "half of the uniform field"
        # The total field contributes H * sum S^z. Split equally: (H/2)*sum S^z each.
        pass
    # field = H * sum_i Sz[i]; each half gets H/2 * sum_i Sz[i].
    for i in range(N_SITE):
        A += (H_FIELD / 2.0) * Sz[i]
        B += (H_FIELD / 2.0) * Sz[i]

    return A, B


# Precompute once (module import is cheap; also cached).
_A, _B = _build_hamiltonians()
_AB = _A + _B


def _expm(mat):
    """Matrix exponential via scipy if available, else numpy fallback."""
    try:
        from scipy.linalg import expm
        return expm(mat)
    except Exception:
        # Numpy-based scaling-and-squaring / eig fallback (adequate for 64x64).
        from numpy.linalg import eig
        w, V = eig(mat)
        return (V @ np.diag(np.exp(w))) @ np.linalg.inv(V)


def _prod_formula(a, b, t):
    """U(t) = prod_{k=1..12} exp(-i b_k t B) exp(-i a_k t A)."""
    U = np.eye(_A.shape[0], dtype=complex)
    for k in range(len(a)):
        U = _expm(-1j * b[k] * t * _B) @ U
        U = _expm(-1j * a[k] * t * _A) @ U
    return U


def _sym_full(ha, hb):
    """Mirror 6-stage half into a 12-stage full palindromic sequence."""
    a = list(ha[:6]) + list(ha[:6][::-1])
    b = list(hb[:6]) + list(hb[:6][::-1])
    return a, b


def _feasy(a, b):
    """Feasibility: sums ~1 and within [-50,50]."""
    if abs(sum(a) - 1.0) > 1e-6 or abs(sum(b) - 1.0) > 1e-6:
        return False
    for x in a + b:
        if not math.isfinite(x) or x < -50.0 or x > 50.0:
            return False
    return True


def _objective(params):
    """Composite objective over the 12-stage palindromic formula.

    params: 12 floats = [ha1..ha6, hb1..hb6]. We mirror to full 12 and
    evaluate spectral norm at task times plus a small L2 norm at dense times.
    """
    ha = params[:6]
    hb = params[6:12]
    a, b = _sym_full(ha, hb)

    # Enforce sum-consistency softly via residual penalty (Nelder-Mead handles
    # the hard feasibility check separately; here we add a small penalty so the
    # search is guided toward sum=1).
    sa = sum(a)
    sb = sum(b)
    pen = (sa - 1.0) ** 2 + (sb - 1.0) ** 2

    val = pen
    for t in T_TASK:
        U = _prod_formula(a, b, t)
        Uex = _expm(-1j * t * _AB)
        val += np.linalg.norm(U - Uex, ord=2)

    # L2 smoothness term at a few extra times to stabilize the landscape.
    for t in (0.1, 0.2, 0.3, 0.7, 0.9):
        U = _prod_formula(a, b, t)
        Uex = _expm(-1j * t * _AB)
        val += 0.1 * np.linalg.norm(U - Uex, ord='fro')

    return val


def coefficients(stages):
    """Return (a, b): 12 finite numbers in [-50,50] with sum(a)=sum(b)=1."""
    deadline = time.time() + 140.0  # explicit guard, 20 s safety margin vs 160 s kill
    stages = int(stages)
    if stages < 12:
        stages = 12

    # Start from the seed (Strang-like) 12-stage solution.
    base_a = [0.0] * 12
    base_b = [0.0] * 12
    # 6 substeps of a simple Strang: per substep a=(1/6,1/6), b=(1/6,0) mirrored.
    n_sub = 6
    for k in range(n_sub):
        base_a[2 * k] = 1.0 / n_sub
        base_a[2 * k + 1] = 1.0 / n_sub
        base_b[2 * k] = 1.0 / n_sub
        base_b[2 * k + 1] = 0.0
    # Mirror to palindromic (keep as-is; it is already symmetric in pairs).
    ha0 = base_a[:6]
    hb0 = base_b[:6]
    params0 = list(ha0) + list(hb0)

    best_params = list(params0)
    best_val = _objective(params0)

    # Nelder-Mead optimization (derivative-free), derivative-free and robust.
    try:
        from scipy.optimize import minimize
        x0 = np.array(params0, dtype=float)
        res = minimize(_objective, x0, method='Nelder-Mead',
                       options={'maxiter': 600, 'fatol': 1e-10, 'xatol': 1e-9,
                                'adaptive': True})
        cand = list(res.x)
        if _objective(cand) < best_val:
            best_val = _objective(cand)
            best_params = cand
        if time.time() > deadline:
            pass
    except Exception:
        pass

    # If we still have time, do a few restarts with perturbed starts.
    rng = np.random.default_rng(0)
    while time.time() < deadline - 30.0:
        pert = best_params + 0.05 * rng.standard_normal(12)
        try:
            from scipy.optimize import minimize
            x0 = np.array(pert, dtype=float)
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'fatol': 1e-9, 'xatol': 1e-9,
                                    'adaptive': True})
            cand = list(res.x)
            v = _objective(cand)
            if v < best_val:
                best_val = v
                best_params = cand
        except Exception:
            break
        if time.time() > deadline:
            break

    # Build final symmetric 12-stage sequences from best params.
    ha = best_params[:6]
    hb = best_params[6:12]
    a, b = _sym_full(ha, hb)

    # Hard feasibility fixups (clip + normalize) while preserving symmetry.
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]

    def normalize(x):
        s = sum(x)
        if abs(s) < 1e-12:
            x[-1] += (1.0 - s)
            return x
        return [v / s for v in x]

    a = normalize(a)
    b = normalize(b)

    # Pad / truncate to exactly `stages` (should be 12).
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    a = a[:stages]
    b = b[:stages]

    # Final clamp after normalization.
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]

    # Re-normalize after clamp to keep sum exactly 1.
    def renorm(x):
        s = sum(x)
        if abs(s) < 1e-12:
            return [1.0 / len(x)] * len(x)
        return [v / s for v in x]
    a = renorm(a)
    b = renorm(b)

    return a, b
