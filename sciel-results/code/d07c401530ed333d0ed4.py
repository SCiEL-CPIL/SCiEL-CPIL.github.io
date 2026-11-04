import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def _build_T(n, m, p):
    NA, NB, NC = n*m, m*p, p*n
    T = np.zeros((NA, NB, NC))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _round_to_rational(arr, max_den=8):
    """Round array elements to nearest rational with small denominator."""
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            best_num, best_den = round(val), 1
            best_err = abs(val - best_num / best_den)
            for den in range(2, max_den + 1):
                num = round(val * den)
                err = abs(val - num / den)
                if err < best_err:
                    best_err = err
                    best_num, best_den = num, den
            result[i, j] = best_num / best_den
    return result

def _als_fit(T, rank, seed, max_iter, deadline):
    NA, NB, NC = T.shape
    rng = np.random.default_rng(seed)
    T1 = T.reshape(NA, NB*NC)
    T2 = T.transpose(1,0,2).reshape(NB, NA*NC)
    T3 = T.transpose(2,0,1).reshape(NC, NA*NB)
    
    # Multiple initialization strategies
    inits = []
    # Random normal
    U = rng.standard_normal((NA, rank))
    V = rng.standard_normal((NB, rank))
    W = rng.standard_normal((NC, rank))
    inits.append((U, V, W))
    
    # Sparse random initialization
    for _ in range(2):
        U_sp = np.zeros((NA, rank))
        V_sp = np.zeros((NB, rank))
        W_sp = np.zeros((NC, rank))
        for r in range(rank):
            U_sp[rng.choice(NA), r] = rng.choice([-1, 1])
            V_sp[rng.choice(NB), r] = rng.choice([-1, 1])
            W_sp[rng.choice(NC), r] = rng.choice([-1, 1])
        inits.append((U_sp, V_sp, W_sp))
    
    def M1(V, W):
        return (V.T[:, :, None] * W.T[:, None, :]).reshape(rank, NB*NC).T
    def M2(U, W):
        return (U.T[:, :, None] * W.T[:, None, :]).reshape(rank, NA*NC).T
    def M3(U, V):
        return (U.T[:, :, None] * V.T[:, None, :]).reshape(rank, NA*NB).T
    
    best_res = np.inf
    best = None
    
    for init_idx, (U_init, V_init, W_init) in enumerate(inits):
        U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
        for it in range(max_iter):
            if time.time() > deadline:
                break
            M = M1(V, W)
            U = np.linalg.lstsq(M, T1.T, rcond=None)[0].T
            M = M2(U, W)
            V = np.linalg.lstsq(M, T2.T, rcond=None)[0].T
            M = M3(U, V)
            W = np.linalg.lstsq(M, T3.T, rcond=None)[0].T
            R = T - np.einsum('ar,br,cr->abc', U, V, W)
            res = np.linalg.norm(R)
            if res < best_res:
                best_res = res
                best = (U.copy(), V.copy(), W.copy())
            if res < 1e-12:
                break
        if best_res < 1e-10:
            break
    
    if best is None:
        best = (U, V, W)
    return best[0], best[1], best[2], best_res

def _try_rounding(U, V, W, T):
    """Try various rounding strategies to get exact integer/half-integer solution."""
    NA, NB, NC = T.shape
    
    # Strategy 1: Round to integers
    for scale in [1, 2, 4]:
        Ur = np.round(U * scale) / scale
        Vr = np.round(V * scale) / scale
        Wr = np.round(W * scale) / scale
        Rr = np.einsum('ar,br,cr->abc', Ur, Vr, Wr) - T
        if np.max(np.abs(Rr)) < 1e-9:
            return Ur, Vr, Wr
    
    # Strategy 2: Round to rationals with small denominators
    for max_den in [2, 4, 8]:
        Ur = _round_to_rational(U, max_den)
        Vr = _round_to_rational(V, max_den)
        Wr = _round_to_rational(W, max_den)
        Rr = np.einsum('ar,br,cr->abc', Ur, Vr, Wr) - T
        if np.max(np.abs(Rr)) < 1e-9:
            return Ur, Vr, Wr
    
    # Strategy 3: Try fixing one factor and rounding others
    for fix_factor in range(3):
        if fix_factor == 0:
            U_fixed = np.round(U)
            # Round V and W
            for scale in [1, 2]:
                Vr = np.round(V * scale) / scale
                Wr = np.round(W * scale) / scale
                Rr = np.einsum('ar,br,cr->abc', U_fixed, Vr, Wr) - T
                if np.max(np.abs(Rr)) < 1e-9:
                    return U_fixed, Vr, Wr
        elif fix_factor == 1:
            V_fixed = np.round(V)
            for scale in [1, 2]:
                Ur = np.round(U * scale) / scale
                Wr = np.round(W * scale) / scale
                Rr = np.einsum('ar,br,cr->abc', Ur, V_fixed, Wr) - T
                if np.max(np.abs(Rr)) < 1e-9:
                    return Ur, V_fixed, Wr
        else:
            W_fixed = np.round(W)
            for scale in [1, 2]:
                Ur = np.round(U * scale) / scale
                Vr = np.round(V * scale) / scale
                Rr = np.einsum('ar,br,cr->abc', Ur, Vr, W_fixed) - T
                if np.max(np.abs(Rr)) < 1e-9:
                    return Ur, Vr, W_fixed
    
    return None

def _try_rank(T, rank, deadline):
    NA, NB, NC = T.shape
    max_iter = 100
    n_seeds = 40
    
    for seed in range(n_seeds):
        if time.time() > deadline - 10:
            break
        U, V, W, res = _als_fit(T, rank, seed, max_iter, deadline - 10)
        if res < 1e-5:
            result = _try_rounding(U, V, W, T)
            if result is not None:
                return result
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_T(n, m, p)
    deadline = time.time() + 130
    
    # Try ranks from low to high
    # Known lower bound for 2x3x9 is around 24-27
    for rank in range(24, n*m*p + 1):
        if time.time() > deadline - 25:
            break
        sol = _try_rank(T, rank, deadline)
        if sol is not None:
            Ur, Vr, Wr = sol
            return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    
    return _schoolbook(n, m, p)
