"""Product-formula coefficient optimizer for 6-site Heisenberg chain.

Strategy: Direct numerical optimization of 24 free coefficients (12 a, 12 b)
with sum constraints enforced via normalization. Uses cached eigen-decompositions
for fast matrix exponential evaluation. Nelder-Mead with multiple restarts
to escape local minima. Deadline guard: return best found before 140s wall-clock.
"""
import numpy as np
import time

_DEADLINE = 140.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------
def _build_hamiltonians():
    N = 6
    delta = 0.8
    field = 0.3

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    def site_op(op, site):
        ops = [I] * N
        ops[site] = op
        return kron_all(ops)

    def bond_op(i, j):
        ops_sz = [I] * N; ops_sz[i] = sz; ops_sz[j] = sz
        ops_sx = [I] * N; ops_sx[i] = sx; ops_sx[j] = sx
        ops_sy = [I] * N; ops_sy[i] = sy; ops_sy[j] = sy
        return delta * (kron_all(ops_sz) + kron_all(ops_sx) + kron_all(ops_sy))

    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)   # even bonds
    A = bond_op(1, 2) + bond_op(3, 4)                   # odd bonds

    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)

    A = A + field_op
    B = B + field_op
    return A, B

# ---------------------------------------------------------------------------
# Cached eigen-decompositions and fast error
# ---------------------------------------------------------------------------
class _EigCache:
    def __init__(self, A, B):
        self.dim = A.shape[0]
        self.I = np.eye(self.dim, dtype=complex)
        self.H = A + B
        self.eig_H, self.vec_H = np.linalg.eigh(self.H)
        self.eig_A, self.vec_A = np.linalg.eigh(A)
        self.eig_B, self.vec_B = np.linalg.eigh(B)
        self.times = [0.25, 0.5, 1.0]
        self.expH = {}
        for t in self.times:
            self.expH[t] = (self.vec_H * np.exp(-1j * t * self.eig_H)) @ self.vec_H.conj().T
        # Precompute eigenvectors for fast exp
        self.vec_A = self.vec_A  # already set
        self.vec_B = self.vec_B  # already set

    def _exp_mat(self, vec, eig, coeff):
        """Compute vec @ diag(exp(-i*coeff*eig)) @ vec^H efficiently."""
        return (vec * np.exp(-1j * coeff * eig)) @ vec.conj().T

    def error(self, a, b):
        """a, b are numpy arrays of length 12 with sums ~1."""
        err_sum = 0.0
        for t in self.times:
            U = self.I
            for k in range(11, -1, -1):
                bk = float(b[k])
                ak = float(a[k])
                if abs(bk) < 1e-14:
                    eb = self.I
                else:
                    eb = self._exp_mat(self.vec_B, self.eig_B, bk * t)
                if abs(ak) < 1e-14:
                    ea = self.I
                else:
                    ea = self._exp_mat(self.vec_A, self.eig_A, ak * t)
                U = eb @ ea @ U
            err_sum += np.linalg.norm(U - self.expH[t], ord=2)
        return err_sum / len(self.times)

# ---------------------------------------------------------------------------
# Direct 24-parameter optimization with normalization
# ---------------------------------------------------------------------------
def _normalize_arrays(a, b):
    """Normalize a and b (numpy arrays) so their sums equal 1, clip to [-50,50]."""
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) < 1e-15:
        a = np.ones(12) / 12.0
        sa = 1.0
    if abs(sb) < 1e-15:
        b = np.ones(12) / 12.0
        sb = 1.0
    a = a / sa
    b = b / sb
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    return a, b

def _st4_initial():
    """Return initial 24-parameter vector: [a_1..a_12, b_1..b_12] from ST4-like values."""
    # Simple equal-split baseline
    a = np.ones(12) / 12.0
    b = np.ones(12) / 12.0
    return np.concatenate([a, b])

def _nelder_mead(objective, x0, max_iter=500, tol=1e-10):
    """Nelder-Mead optimizer with deadline guard."""
    n = len(x0)
    x0 = np.asarray(x0, dtype=float)
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if abs(xi[i]) < 1e-8 else 0.1 * np.sign(xi[i])
        xi[i] += step
        simplex.append(xi)
    fvals = [objective(x) for x in simplex]

    for it in range(max_iter):
        if _remaining() < 10:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if fvals[-1] - fvals[0] < tol:
            break

        centroid = np.mean(simplex[:n], axis=0)
        x_ref = centroid + 1.0 * (centroid - simplex[n])
        f_ref = objective(x_ref)

        if f_ref < fvals[0]:
            x_exp = centroid + 2.0 * (centroid - simplex[n])
            f_exp = objective(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp; fvals[n] = f_exp
            else:
                simplex[n] = x_ref; fvals[n] = f_ref
        elif f_ref < fvals[n-1]:
            simplex[n] = x_ref; fvals[n] = f_ref
        else:
            x_con = centroid + 0.5 * (simplex[n] - centroid)
            f_con = objective(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con; fvals[n] = f_con
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    best_idx = int(np.argmin(fvals))
    return simplex[best_idx], fvals[best_idx]

def coefficients(stages):
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0); b.append(0.0)
        a = a[:stages]; b = b[:stages]
        return a, b

    A, B = _build_hamiltonians()
    cache = _EigCache(A, B)

    def objective(params):
        """params: 24-element array [a_1..a_12, b_1..b_12]."""
        a = np.asarray(params[:12], dtype=float)
        b = np.asarray(params[12:], dtype=float)
        a, b = _normalize_arrays(a, b)
        return cache.error(a, b)

    # Multiple random restarts
    np.random.seed(42)
    best_params = _st4_initial()
    best_err = objective(best_params)

    n_restarts = 8
    for r in range(n_restarts):
        if _remaining() < 20:
            break
        if r == 0:
            x0 = _st4_initial()
        else:
            # Perturb from best so far
            x0 = best_params + np.random.randn(24) * 0.05
        params_opt, f_opt = _nelder_mead(objective, x0, max_iter=300)
        if f_opt < best_err:
            best_err = f_opt
            best_params = params_opt

    # Final polish with tighter tolerance if time allows
    if _remaining() > 30:
        params_polish, f_polish = _nelder_mead(objective, best_params, max_iter=500, tol=1e-12)
        if f_polish < best_err:
            best_err = f_polish
            best_params = params_polish

    a_opt = np.asarray(best_params[:12], dtype=float)
    b_opt = np.asarray(best_params[12:], dtype=float)
    a_opt, b_opt = _normalize_arrays(a_opt, b_opt)

    # Convert to lists
    return list(a_opt), list(b_opt)
