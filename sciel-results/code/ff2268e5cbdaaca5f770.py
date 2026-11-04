"""Direct algebraic optimization for 4x4x7 tensor rank minimization."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    U[0, 0] = 1; U[3, 0] = -1
    V[3, 0] = 1; V[0, 0] = -1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1
    U[0, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    U[3, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    U[0, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1; W[3, 6] = 1
    return U, V, W


def _base_block_decomp(n, m, p):
    """Block decomposition: Strassen 2x2x2 for 4x4x4 part, schoolbook for 4x4x3."""
    r_total = 8 * 7 + 4 * 12  # 104
    U = np.zeros((n * m, r_total), dtype=np.float64)
    V = np.zeros((m * p, r_total), dtype=np.float64)
    W = np.zeros((p * n, r_total), dtype=np.float64)
    sU, sV, sW = _strassen_2x2x2()
    blocks = [(0,0,0,0),(0,1,1,0),(1,0,0,0),(1,1,1,0),(0,0,0,1),(0,1,1,1),(1,0,0,1),(1,1,1,1)]
    off = 0
    for (arb, acb, brb, bc) in blocks:
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[(2*arb+i)*4 + (2*acb+j), off+r] += sU[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[(2*brb+j)*7 + (2*bc+k), off+r] += sV[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[(2*bc+k)*4 + (2*arb+i), off+r] += sW[k*2+i, r]
        off += 7
    for (arb, acb, brb) in [(0,0,0),(0,1,1),(1,0,0),(1,1,1)]:
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    col = off + i*6 + j*3 + k
                    U[(2*arb+i)*4 + (2*acb+j), col] = 1.0
                    V[(2*brb+j)*7 + (4+k), col] = 1.0
                    W[(4+k)*4 + (2*arb+i), col] = 1.0
        off += 12
    return U, V, W


def _snap_halfint(X):
    return (2.0 * np.round(X * 0.5)) / 2.0


def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    r = U.shape[1]
    Uf = [[Fraction(int(round(U[i, j] * 2))) / 2 for j in range(r)] for i in range(n*m)]
    Vf = [[Fraction(int(round(V[i, j] * 2))) / 2 for j in range(r)] for i in range(m*p)]
    Wf = [[Fraction(int(round(W[i, j] * 2))) / 2 for j in range(r)] for i in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for rr in range(r):
                    s += Uf[i*m+j][rr] * Vf[j*p+k][rr] * Wf[k*n+i][rr]
                if s != 1:
                    return False
    return True


def _verify_float(U, V, W, n, m, p):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _optimize(U0, V0, W0, n, m, p, deadline, seed=0):
    """L-BFGS-B optimization of rank-r factors, with periodic half-int snap + exact verify."""
    from scipy.optimize import minimize
    rng = np.random.RandomState(seed)
    r = U0.shape[1]
    nm, mp, pn = n*m, m*p, p*n

    # Flatten with small random perturbation to escape the block structure
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    # Small perturbation to break exactness so optimizer can move
    U += rng.normal(0, 0.01, U.shape)
    V += rng.normal(0, 0.01, V.shape)
    W += rng.normal(0, 0.01, W.shape)

    def unpack(x):
        Ux = x[:nm*r].reshape(nm, r)
        Vx = x[nm*r:nm*r+mp*r].reshape(mp, r)
        Wx = x[nm*r+mp*r:].reshape(pn, r)
        return Ux, Vx, Wx

    def loss_grad(x):
        Ux, Vx, Wx = unpack(x)
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Residual: R[i,j,k] = T[i,j,k] - 1
        # grad_U[i*m+j, r] = sum_k V[j*p+k,r] * W[k*n+i,r] * (T[i,j,k]-1)
        # grad_V[j*p+k, r] = sum_i U[i*m+j,r] * W[k*n+i,r] * (T[i,j,k]-1)
        # grad_W[k*n+i, r] = sum_j U[i*m+j,r] * V[j*p+k,r] * (T[i,j,k]-1)
        T = np.zeros((n, m, p), dtype=np.float64)
        for rr in range(r):
            # U[:,rr] is (nm,), V[:,rr] is (mp,), W[:,rr] is (pn,)
            # T[i,j,k] += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
            Uv = Ux[:, rr].reshape(n, m)
            Vv = Vx[:, rr].reshape(m, p)
            Wv = Wx[:, rr].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += Uv[i, j] * Vv[j, k] * Wv[k, i]
        R = T - 1.0
        loss = 0.5 * np.sum(R * R)

        gU = np.zeros_like(Ux)
        gV = np.zeros_like(Vx)
        gW = np.zeros_like(Wx)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Rijk = R[i, j, k]
                    for rr in range(r):
                        Uv = Ux[i*m+j, rr]
                        Vv = Vx[j*p+k, rr]
                        Wv = Wx[k*n+i, rr]
                        gU[i*m+j, rr] += Rijk * Vv * Wv
                        gV[j*p+k, rr] += Rijk * Uv * Wv
                        gW[k*n+i, rr] += Rijk * Uv * Vv
        grad = np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
        return loss, grad

    x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])

    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_loss = np.inf

    # If the base is already exact on half-int grid, record it
    if _verify_exact(U0, V0, W0, n, m, p):
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_loss = 0.0

    t_start = time.time()
    # Run optimizer with maxiter limited; check deadline between iterations via callback
    state = {'count': 0, 'stop': False}

    def cb(xk):
        state['count'] += 1
        if state['count'] % 50 == 0:
            if time.time() > deadline:
                state['stop'] = True
        return state['stop']

    try:
        res = minimize(loss_grad, x0, method='L-BFGS-B', jac=True,
                       callback=cb,
                       options={'maxiter': 2000, 'maxfun': 50000,
                                'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    except Exception:
        res = None

    if res is not None and hasattr(res, 'x') and res.x is not None:
        Uo, Vo, Wo = unpack(res.x)
        # Snap to half-int
        Us, Vs, Ws = _snap_halfint(Uo), _snap_halfint(Vo), _snap_halfint(Wo)
        # Try exact verify on snapped
        if _verify_exact(Us, Vs, Ws, n, m, p):
            best_U, best_V, best_W = Us, Vs, Ws
            best_loss = 0.0
        else:
            # Try unsnap verify (in case snap broke it but original is close)
            if _verify_float(Uo, Vo, Wo, n, m, p):
                # snap again more carefully
                Us2 = np.round(Uo * 2) / 2
                Vs2 = np.round(Vo * 2) / 2
                Ws2 = np.round(Wo * 2) / 2
                if _verify_exact(Us2, Vs2, Ws2, n, m, p):
                    best_U, best_V, best_W = Us2, Vs2, Ws2
                    best_loss = 0.0

    # Local search: try removing columns that are linearly dependent
    # and re-verify
    if best_loss > 0 and time.time() < deadline - 5:
        # Try to reduce rank by dropping a column and re-optimizing
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            r_cur = best_U.shape[1]
            if r_cur <= 1:
                break
            # Try dropping each column, re-optimize, check exactness
            improved = False
            for drop in range(r_cur):
                if time.time() > deadline - 10:
                    break
                U_d = np.delete(best_U, drop, axis=1)
                V_d = np.delete(best_V, drop, axis=1)
                W_d = np.delete(best_W, drop, axis=1)
                # Quick float check
                if not _verify_float(U_d, V_d, W_d, n, m, p):
                    continue
                if _verify_exact(U_d, V_d, W_d, n, m, p):
                    best_U, best_V, best_W = U_d, V_d, W_d
                    best_loss = 0.0
                    improved = True
                    break
            if not improved:
                break

    return best_U, best_V, best_W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Start from block decomposition (rank 104)
    U, V, W = _base_block_decomp(n, m, p)

    # Verify base
    if not _verify_exact(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)

    # Optimize to reduce rank / find exact half-int decomposition
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            U, V, W = _optimize(U, V, W, n, m, p, deadline, seed=42)
        except Exception:
            pass

    # If we still have time, try a second optimization with different seed
    remaining = deadline - time.time()
    if remaining > 30:
        try:
            U2, V2, W2 = _optimize(U, V, W, n, m, p, deadline, seed=123)
            if U2.shape[1] < U.shape[1]:
                U, V, W = U2, V2, W2
        except Exception:
            pass

    # Final exact verification
    if not _verify_exact(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)

    # Convert to lists
    def _to_list(arr):
        # Values are half-integers; return as floats (0.5, 1.0, -1.0, etc.)
        return arr.astype(np.float64).tolist()

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
