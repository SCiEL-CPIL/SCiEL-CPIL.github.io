import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    nr, mr, pr = n * m, m * p, p * n  # 4, 16, 16

    # Build the multiplication tensor T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nr, mr, pr), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-8):
        Tc = np.einsum('at,bt,ct->abc', U, V, W)
        err = float(np.max(np.abs(Tc - T)))
        return err < tol, err

    # Rank-32 schoolbook fallback
    def get_schoolbook_32():
        U = np.zeros((nr, 32)); V = np.zeros((mr, 32)); W = np.zeros((pr, 32))
        for idx in range(32):
            i = idx // 16; j = (idx // 8) % 2; k = idx % 8
            U[i * m + j, idx] = 1.0
            V[j * p + k, idx] = 1.0
            W[k * n + i, idx] = 1.0
        return U, V, W

    # Rank-28 Strassen fallback
    def get_strassen_28():
        rank = 28
        U = np.zeros((nr, rank), dtype=float)
        V = np.zeros((mr, rank), dtype=float)
        W = np.zeros((pr, rank), dtype=float)
        strassen_products = [
            {'A': {(0, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1, (1, 1): 1}, 'C': {(0, 0): 1, (1, 1): 1}},
            {'A': {(1, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1}, 'C': {(1, 0): 1, (1, 1): -1}},
            {'A': {(0, 0): 1}, 'B': {(0, 1): 1, (1, 1): -1}, 'C': {(0, 1): 1, (1, 1): 1}},
            {'A': {(1, 1): 1}, 'B': {(1, 0): 1, (0, 0): -1}, 'C': {(0, 0): 1, (1, 0): 1}},
            {'A': {(0, 0): 1, (0, 1): 1}, 'B': {(1, 1): 1}, 'C': {(0, 0): -1, (0, 1): 1}},
            {'A': {(1, 0): 1, (0, 0): -1}, 'B': {(0, 0): 1, (0, 1): 1}, 'C': {(1, 1): 1}},
            {'A': {(0, 1): 1, (1, 1): -1}, 'B': {(1, 0): 1, (1, 1): 1}, 'C': {(0, 0): 1}},
        ]
        for q in range(p // 2):
            k1 = 2 * q
            for r, prod in enumerate(strassen_products):
                t = q * 7 + r
                for (i, j), coeff in prod['A'].items():
                    U[i * m + j, t] = coeff
                for (j, k_sub), coeff in prod['B'].items():
                    V[j * p + (k1 + k_sub), t] = coeff
                for (i, k_sub), coeff in prod['C'].items():
                    W[(k1 + k_sub) * n + i, t] = coeff
        return U, V, W

    def round_halfint(U, V, W):
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if verify(U2, V2, W2, tol=1e-7)[0]:
            return U2, V2, W2
        Ui = np.round(U); Vi = np.round(V); Wi = np.round(W)
        if verify(Ui, Vi, Wi, tol=1e-7)[0]:
            return Ui, Vi, Wi
        return None

    def polish_lbfgs(U0, V0, W0, deadline, tol=1e-10, maxiter=200):
        U_flat = U0.flatten()
        V_flat = V0.flatten()
        W_flat = W0.flatten()
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        nU = U0.size
        nV = V0.size

        def loss_and_grad(x):
            U = x[:nU].reshape(nr, -1)
            V = x[nU:nU+nV].reshape(mr, -1)
            W = x[nU+nV:].reshape(pr, -1)
            err = np.einsum('at,bt,ct->abc', U, V, W) - T
            loss = float(np.sum(err * err))
            gU = np.einsum('abc,bt,ct->at', err, V, W)
            gV = np.einsum('abc,at,ct->bt', err, U, W)
            gW = np.einsum('abc,at,bt->ct', err, U, V)
            grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            return loss, grad

        try:
            res = minimize(
                loss_and_grad, x0, jac=True, method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8}
            )
            x_opt = res.x
            U_opt = x_opt[:nU].reshape(nr, -1)
            V_opt = x_opt[nU:nU+nV].reshape(mr, -1)
            W_opt = x_opt[nU+nV:].reshape(pr, -1)
            err = np.einsum('at,bt,ct->abc', U_opt, V_opt, W_opt) - T
            loss = float(np.sum(err * err))
            return U_opt, V_opt, W_opt, loss
        except Exception:
            return U0, V0, W0, np.inf

    start = time.time()
    deadline = start + 140.0

    # Always-valid baselines
    U28, V28, W28 = get_strassen_28()
    ok28, _ = verify(U28, V28, W28)
    if not ok28:
        U, V, W = get_schoolbook_32()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 28
    best_U, best_V, best_W = U28, V28, W28

    # Try rank 24: exploit polynomial structure
    # Interpret 8 columns as coefficients of degree-3 polynomial in 2 vars
    # Search for rank-24 by finding algebraic dependencies
    for target in [24, 25, 26, 27]:
        if time.time() > deadline - 10.0:
            break
        found = False
        for restart in range(8):
            if time.time() > deadline - 5.0:
                break
            rng = np.random.default_rng(100000 + target * 13 + restart * 7)
            
            # Structured init: start from rank-28 and perturb
            U0, V0, W0 = U28[:, :target].copy(), V28[:, :target].copy(), W28[:, :target].copy()
            U0 += rng.normal(0, 0.1, size=U0.shape)
            V0 += rng.normal(0, 0.1, size=V0.shape)
            W0 += rng.normal(0, 0.1, size=W0.shape)
            
            U1, V1, W1, loss1 = polish_lbfgs(U0, V0, W0, deadline, tol=1e-10, maxiter=150)
            
            if loss1 < 1e-9:
                r = round_halfint(U1, V1, W1)
                if r is not None:
                    best_rank = target
                    best_U, best_V, best_W = r
                    found = True
                    break
        
        if found:
            continue
        break

    U, V, W = best_U, best_V, best_W
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
