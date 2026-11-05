"""Optimize fixed 3-layer Ry/CZ circuit for open transverse-field Ising ground states."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize as opt

# ---- exact ground energy (independent reference) ----
def _ground_energy(n, coupling, field):
    dim = 1 << n
    I = np.eye(2); X = np.array([[0,1],[1,0]]); Z = np.array([[1,0],[0,-1]])
    def kron_all(ops):
        r = ops[0]
        for o in ops[1:]:
            r = np.kron(r, o)
        return r
    H = np.zeros((dim, dim), dtype=float)
    for i in range(n-1):
        ops = [I]*n; ops[i] = Z; ops[i+1] = Z
        H -= coupling * kron_all(ops)
    for i in range(n):
        ops = [I]*n; ops[i] = X
        H -= field * kron_all(ops)
    return float(np.linalg.eigvalsh(H)[0])

# ---- circuit state evolution ----
def _state_from_params(theta, n):
    L = theta.shape[0]
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    for l in range(L):
        t = theta[l]
        for i in range(n):
            c = np.cos(0.5*t[i]); s = np.sin(0.5*t[i])
            bit = 1 << i
            nbits = 1 << n
            for b in range(nbits):
                if (b & bit) == 0:
                    psi[b] = c*psi[b] - s*psi[b|bit]
                else:
                    psi[b] = s*psi[b^bit] + c*psi[b]
        if l < L-1:
            for i in range(n-1):
                b1 = 1 << i; b2 = 1 << (i+1)
                for b in range(1 << n):
                    if (b & b1) and (b & b2):
                        psi[b] = -psi[b]
    return psi

# ---- local single-qubit operator (dense, reused) ----
def _local_op(n, i, M):
    ops = [np.eye(2, dtype=complex)]*n
    ops[i] = M
    r = ops[0]
    for o in ops[1:]:
        r = np.kron(r, o)
    return r

# ---- energy + analytic gradient ----
def _energy_and_grad(theta, n, coupling, field):
    L = theta.shape[0]
    psi = _state_from_params(theta, n)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    OX = [_local_op(n,i,X) for i in range(n)]
    OZ = [_local_op(n,i,Z) for i in range(n)]
    def expv(O):
        return np.real(np.vdot(psi, O @ psi))
    energy = 0.0
    for i in range(n-1):
        energy -= coupling * expv(OZ[i] @ OZ[i+1])
    for i in range(n):
        energy -= field * expv(OX[i])
    grad = np.zeros_like(theta)
    for l in range(L):
        t = theta[l]
        for i in range(n):
            st = np.sin(0.5*t[i]); ct = np.cos(0.5*t[i])
            if st == 0.0:
                continue
            Zt = -0.5*st * OZ[i]
            g = 0.0
            for j in range(n-1):
                if i == j or i == j+1:
                    g -= coupling * expv(OZ[j] @ Zt + Zt @ OZ[j+1])
            g -= field * expv(OX[i] @ Zt + Zt @ OX[i])
            grad[l, i] = g
    return energy, grad

# ---- objective (mean gap per spin) ----
def _objective(theta, n, coupling, field, ge):
    e, g = _energy_and_grad(theta, n, coupling, field)
    return (e - ge)/n, g/n

def _initial_theta(n, field):
    L = 3
    theta = np.zeros((L, n))
    if field > 0.0:
        theta[0] = np.pi/2
        theta[1] = np.pi/2
        theta[2] = np.arccos(1.0/(1.0 + field*field))
    return theta

def _optimize_one(n, coupling, field, deadline, start):
    ge = _ground_energy(n, coupling, field)
    best = None
    for restart in range(3):
        if time.time() > deadline:
            break
        if restart == 0:
            x0 = _initial_theta(n, field)
        else:
            x0 = np.random.randn(3, n) * 0.3
        try:
            res = opt.minimize(_objective, x0, args=(n, coupling, field, ge),
                               method='L-BFGS-B', jac=True,
                               options={'maxiter': 80, 'maxfun': 2000, 'ftol': 1e-12})
            if res.success or res.fun < 1e12:
                val = _objective(res.x, n, coupling, field, ge)[0]
                if best is None or val < best[0]:
                    best = (val, res.x.copy())
        except Exception:
            pass
    if best is None:
        return _initial_theta(n, field)
    return best[1]

def angles(n, coupling, field, layers):
    start = time.time()
    deadline = start + 145.0
    dev_fields = [0.3, 0.8, 1.4]
    # warm the optimizer on dev fields (also gives good initial structure)
    for f in dev_fields:
        if time.time() > deadline:
            break
        _optimize_one(n, coupling, f, deadline, start)
    # final answer for the requested field
    if time.time() > deadline:
        return _initial_theta(n, field)
    return _optimize_one(n, coupling, field, deadline, start)
# EVOLVE-BLOCK-END
