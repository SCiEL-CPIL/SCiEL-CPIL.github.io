import numpy as np
import time
import random
from functools import lru_cache
from itertools import product

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i  # W[k*n+i]

def verify_np(U, V, W):
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    result = np.einsum('ijl,jkl,kil->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=1e-9)

def baseline_66():
    U = [[0]*66 for _ in range(n*m)]
    V = [[0]*66 for _ in range(m*p)]
    W = [[0]*66 for _ in range(p*n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
        A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
        C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
        U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
        U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
        U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
        U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
        U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
        U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
        U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
        U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
        U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def verify_np(U, V, W):
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    result = np.einsum('ijl,jkl,kil->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=1e-9)

def build_from_terms(terms):
    """Build U, V, W from a list of (i,j,k) rank-1 terms."""
    U = np.zeros((n*m, len(terms)), dtype=float)
    V = np.zeros((m*p, len(terms)), dtype=float)
    W = np.zeros((p*n, len(terms)), dtype=float)
    for l, (i, j, k) in enumerate(terms):
        U[Aidx(i, j), l] = 1.0
        V[Bidx(j, k), l] = 1.0
        W[Cidx(i, k), l] = 1.0
    return U, V, W

def dp_solve(i0, i1, j0, j1, k0, k1, deadline):
    """Solve sub-problem: multiply A[i0:i1, j0:j1] by B[j0:j1, k0:k1].
    Returns list of (i,j,k) terms for the sub-problem."""
    rows = i1 - i0
    cols = k1 - k0
    mid = j1 - j0
    
    # Base case: 1x1 sub-problem
    if rows == 1 and cols == 1 and mid == 1:
        return [(i0, j0, k0)]
    
    # Try splitting along rows (A dimension)
    best = None
    best_rank = rows * cols * mid  # naive upper bound
    
    # Split A rows
    for split in range(1, rows):
        if time.time() > deadline:
            break
        terms1 = dp_solve(i0, i0+split, j0, j1, k0, k1, deadline)
        terms2 = dp_solve(i0+split, i1, j0, j1, k0, k1, deadline)
        combined = terms1 + terms2
        if len(combined) < best_rank:
            best_rank = len(combined)
            best = combined
    
    # Split B columns (k dimension)
    for split in range(1, cols):
        if time.time() > deadline:
            break
        terms1 = dp_solve(i0, i1, j0, j1, k0, k0+split, deadline)
        terms2 = dp_solve(i0, i1, j0, j1, k0+split, k1, deadline)
        combined = terms1 + terms2
        if len(combined) < best_rank:
            best_rank = len(combined)
            best = combined
    
    # Split B rows / A cols (j dimension)
    for split in range(1, mid):
        if time.time() > deadline:
            break
        terms1 = dp_solve(i0, i1, j0, j0+split, k0, k1, deadline)
        terms2 = dp_solve(i0, i1, j0+split, j1, k0, k1, deadline)
        combined = terms1 + terms2
        if len(combined) < best_rank:
            best_rank = len(combined)
            best = combined
    
    if best is None:
        # Fallback: naive
        best = [(i, j, k) for i in range(i0, i1) for j in range(j0, j1) for k in range(k0, k1)]
    
    return best

def try_drop(U, V, W, l):
    U_new = np.delete(U, l, axis=1)
    V_new = np.delete(V, l, axis=1)
    W_new = np.delete(W, l, axis=1)
    return verify_np(U_new, V_new, W_new)

def local_search(U, V, W, deadline, rng):
    r = U.shape[1]
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        indices = list(range(r))
        rng.shuffle(indices)
        for l in indices:
            if time.time() > deadline - 10:
                break
            if try_drop(U, V, W, l):
                U = np.delete(U, l, axis=1)
                V = np.delete(V, l, axis=1)
                W = np.delete(W, l, axis=1)
                r -= 1
                improved = True
                break
    return U, V, W

def construct():
    deadline = time.time() + 130
    rng = random.Random(42)
    
    # Try DP-based decomposition
    best_U, best_V, best_W = baseline_66()
    best_rank = 66
    
    try:
        if time.time() < deadline - 30:
            terms = dp_solve(0, n, 0, m, 0, p, deadline - 20)
            U_dp, V_dp, W_dp = build_from_terms(terms)
            if verify_np(U_dp, V_dp, W_dp) and len(terms) < best_rank:
                best_U, best_V, best_W = U_dp, V_dp, W_dp
                best_rank = len(terms)
    except Exception:
        pass
    
    # Apply local search to best result
    U, V, W = local_search(best_U.copy(), best_V.copy(), best_W.copy(), deadline, rng)
    
    if not verify_np(U, V, W):
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
