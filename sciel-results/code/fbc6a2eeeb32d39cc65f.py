import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 11]
DEADLINE = 150.0
SAFETY = 5.0  # seconds to leave before hard kill

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS

    # --- Guaranteed rank-39 construction (Strassen blocks + schoolbook tail) ---
    def build_rank39():
        rank = 5 * 7 + 1 * 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0; v_f = k1; v_g = p + k0; v_h = p + k1
            w_c00 = k0 * n; w_c10 = k0 * n + 1
            w_c01 = k1 * n; w_c11 = k1 * n + 1
            # m1
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            # m2
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            # m3
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            # m4
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            # m5
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            # m6
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            # m7
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        # single column k2 = 10, schoolbook 4
        k2 = 10
        v_e2 = k2; v_g2 = p + k2
        w_c02 = k2 * n; w_c12 = k2 * n + 1
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        return {"U": U, "V": V, "W": W}

    best = build_rank39()

    # --- Verification helpers ---
    def verify(U, V, W):
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        # T shape: (n*m, m*p, p*n)
        T = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        
        # Construct target tensor M of shape (n*m, m*p, p*n)
        # M[a, b, c] = 1 if indices are consistent, else 0
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        M = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    M[a, b, c] = 1.0
        
        return np.allclose(T, M, atol=1e-9, rtol=0.0)

    def round_halfint(mat):
        return np.round(np.asarray(mat, dtype=float) * 2.0) / 2.0

    def build_target_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = build_target_tensor()
    T_flat = T_target.flatten()
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    def loss(params, rank):
        U = params[:dim_u * rank].reshape(dim_u, rank)
        V = params[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
        W = params[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((T_rec.flatten() - T_flat) ** 2)

    def try_rank(rank, n_restarts, maxiter, seed_base=123):
        if time.time() - t_start > DEADLINE - 15:
            return None
        
        rng = np.random.default_rng(seed_base)
        best_loss = float('inf')
        best_params = None
        
        for idx in range(n_restarts):
            if time.time() - t_start > DEADLINE - 10:
                break
            U_init = rng.normal(0, 1, (dim_u, rank))
            V_init = rng.normal(0, 1, (dim_v, rank))
            W_init = rng.normal(0, 1, (dim_w, rank))
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            res = minimize(loss, params_init, args=(rank,), method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
            if best_loss < 1e-8:
                break

        if best_params is None or best_loss > 1e-6:
            return None

        U = best_params[:dim_u * rank].reshape(dim_u, rank)
        V = best_params[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
        W = best_params[dim_u * rank + dim_v * rank:].reshape(dim_w, rank)
        U_r = round_halfint(U)
        V_r = round_halfint(V)
        W_r = round_halfint(W)
        
        if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        return None

    # Try rank 38
    if time.time() - t_start < DEADLINE - 20:
        r38 = try_rank(38, n_restarts=15, maxiter=200, seed_base=123)
        if r38:
            return r38

    # Try rank 37
    if time.time() - t_start < DEADLINE - 10:
        r37 = try_rank(37, n_restarts=10, maxiter=150, seed_base=456)
        if r37:
            return r37

    return best
