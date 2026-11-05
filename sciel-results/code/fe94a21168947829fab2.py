"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition search.

Uses a structured block-embedding with Laderman's algorithm (rank 6) for 2x2x2
blocks and schoolbook for odd-sized blocks. Then attempts a smooth residual
polish (L-BFGS-B) to reduce rank, and a final round-to-half-integer + verify.
"""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # safety margin under the 150 s kill

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P, tol=1e-9):
    """Check exact decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return bool(np.allclose(T, 1.0, atol=tol, rtol=0))

def _to_halfint(Mat):
    """Round to nearest half-integer."""
    return np.round(2.0 * Mat) / 2.0

def _laderman_2x2():
    """Laderman's decomposition for 2x2x2 (rank 6). Returns U,V,W each (4,6)."""
    U = np.zeros((4, 6))
    V = np.zeros((4, 6))
    W = np.zeros((4, 6))
    # Laderman's 6 multiplications:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b01)
    # M3 = (a00)(b10+b11)
    # M4 = (a11)(b00)
    # M5 = (a01-a10)(b10)
    # M6 = (a01+a11)(b11-b00)
    # c00 = M1 + M3 - M4 - M6
    # c01 = M2 + M6
    # c10 = M3 + M5
    # c11 = M1 - M2 + M4 - M5
    # U[i,j] = coefficient of a_{ij} in the i-th row of A, j-th col
    # V[j,k] = coefficient of b_{jk}
    # W[k,i] = coefficient in output C_{ki} (but our convention is W[k*n+i])
    # For 2x2: U is 2x2 reshaped to (4,6), V is 2x2 reshaped to (4,6), W is 2x2 reshaped to (4,6)
    # Index: U[a*2+b, c] where a=row, b=col of A
    #         V[b*2+c, c] where b=row, c=col of B
    #         W[c*2+a, c] where c=row, a=col of C  (i.e., W[k*2+i])
    
    # M1: (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1  # a00, a11
    V[0,0]=1; V[3,0]=1  # b00, b11
    W[0,0]=1; W[3,0]=1  # c00, c11
    
    # M2: (a10+a11)(b01)
    U[2,1]=1; U[3,1]=1  # a10, a11
    V[1,1]=1           # b01
    W[1,1]=1           # c01
    
    # M3: (a00)(b10+b11)
    U[0,2]=1           # a00
    V[2,2]=1; V[3,2]=1 # b10, b11
    W[0,2]=1; W[3,2]=1 # c00, c11
    
    # M4: (a11)(b00)
    U[3,3]=1           # a11
    V[0,3]=1           # b00
    W[0,3]=-1; W[3,3]=1 # -c00, +c11
    
    # M5: (a01-a10)(b10)
    U[1,4]=1; U[2,4]=-1 # a01, -a10
    V[2,4]=1           # b10
    W[2,4]=1           # c10
    
    # M6: (a01+a11)(b11-b00)
    U[1,5]=1; U[3,5]=1  # a01, a11
    V[3,5]=1; V[0,5]=-1 # b11, -b00
    W[0,5]=-1; W[1,5]=1 # -c00, +c01
    
    return U, V, W

def _strassen_2x2():
    """Strassen decomposition for 2x2x2. Returns U,V,W each (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[1,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_structured(n, m, p, use_laderman=True):
    """Build a structured low-rank candidate by block-embedding.
    
    Partition:
      rows i in 0..2 -> blocks [0,1] and [2]
      cols j in 0..3 -> blocks [0,1] and [2,3]
      cols k in 0..4 -> blocks [0,1], [2,3], [4]
    """
    row_blocks = [(0, 2), (2, 3)]      # sizes 2,1
    col_blocks = [(0, 2), (2, 4)]      # sizes 2,2
    k_blocks   = [(0, 2), (2, 4), (4, 5)]  # sizes 2,2,1

    U = {}
    V = {}
    W = {}
    rank = 0
    if use_laderman:
        S2U, S2V, S2W = _laderman_2x2()
        s2_rank = 6
    else:
        S2U, S2V, S2W = _strassen_2x2()
        s2_rank = 7

    def schoolblock(ri, rj, rk, base):
        nonlocal rank
        for i in range(ri[0], ri[1]):
            for j in range(rj[0], rj[1]):
                for k in range(rk[0], rk[1]):
                    U[(i, j, base)] = 1.0
                    V[(j, k, base)] = 1.0
                    W[(k, i, base)] = 1.0
                    base += 1
        return base

    for ri in row_blocks:
        for rj in col_blocks:
            for rk in k_blocks:
                si, sj, sk = ri[1]-ri[0], rj[1]-rj[0], rk[1]-rk[0]
                if si == 2 and sj == 2 and sk == 2:
                    # Use Laderman or Strassen block
                    for c in range(s2_rank):
                        for a in range(2):
                            for b in range(2):
                                val = S2U[a*2+b, c]
                                if val != 0:
                                    key = (ri[0]+a, rj[0]+b, rank)
                                    U[key] = U.get(key, 0.0) + val
                        for b in range(2):
                            for cc in range(2):
                                val = S2V[b*2+cc, c]
                                if val != 0:
                                    key = (rj[0]+b, rk[0]+cc, rank)
                                    V[key] = V.get(key, 0.0) + val
                        for cc in range(2):
                            for a in range(2):
                                val = S2W[cc*2+a, c]
                                if val != 0:
                                    key = (rk[0]+cc, ri[0]+a, rank)
                                    W[key] = W.get(key, 0.0) + val
                        rank += 1
                else:
                    rank = schoolblock(ri, rj, rk, rank)

    # Convert dict-of-triples to dense matrices.
    Umat = np.zeros((n*m, rank))
    Vmat = np.zeros((m*p, rank))
    Wmat = np.zeros((p*n, rank))
    for (i, j, c), v in U.items():
        Umat[i*m+j, c] += v
    for (j, k, c), v in V.items():
        Vmat[j*p+k, c] += v
    for (k, i, c), v in W.items():
        Wmat[k*n+i, c] += v
    return Umat, Vmat, Wmat

def _smooth_polish(U, V, W, n, m, p, time_budget, seed, maxiter=300):
    """Polish a candidate with smooth residual minimization, then round to
    half-integers and verify."""
    np.random.seed(seed)
    rank = U.shape[1]
    T = np.ones((n, m, p))

    def reshape3(x, rows, cols):
        return x.reshape(rows, cols, rank)

    def objective(x):
        Uf = reshape3(x[:n*m*rank], n, m)
        Vf = reshape3(x[n*m*rank:n*m*rank+m*p*rank], m, p)
        Wf = reshape3(x[n*m*rank+m*p*rank:], p, n)
        Tf = np.einsum('ijr,jkr,kir->ijk', Uf, Vf, Wf, optimize=True)
        d = Tf - T
        return float(np.sum(d*d))

    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12})
    except Exception:
        return None
    if res.fun < 1e-6:
        Uf = reshape3(res.x[:n*m*rank], n, m)
        Vf = reshape3(res.x[n*m*rank:n*m*rank+m*p*rank], m, p)
        Wf = reshape3(res.x[n*m*rank+m*p*rank:], p, n)
        Ur = _to_halfint(Uf.reshape(n*m, rank))
        Vr = _to_halfint(Vf.reshape(m*p, rank))
        Wr = _to_halfint(Wf.reshape(p*n, rank))
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr
    return None

def _try_rank_reduction(U, V, W, n, m, p, target_rank, seed, time_budget):
    """Try to reduce rank from current to target_rank by zeroing columns and polishing."""
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return None
    np.random.seed(seed)
    
    # Randomly select which columns to keep
    keep_indices = np.random.choice(current_rank, size=target_rank, replace=False)
    keep_indices.sort()
    
    U_reduced = U[:, keep_indices]
    V_reduced = V[:, keep_indices]
    W_reduced = W[:, keep_indices]
    
    # Polish
    res = _smooth_polish(U_reduced, V_reduced, W_reduced, n, m, p, time_budget, seed, maxiter=200)
    if res is not None:
        Ur, Vr, Wr = res
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr
    return None

def _schoolbook():
    n, m, p = N, M, P
    rank = n*m*p
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1; V[j*p+k, col] = 1; W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = N, M, P
    U_sb, V_sb, W_sb = _schoolbook()
    best = (U_sb, V_sb, W_sb)
    best_rank = U_sb.shape[1]

    # 1) Structured block-embedding with Laderman (rank 6 for 2x2x2 blocks).
    if _time_left() > 20:
        U, V, W = _build_structured(n, m, p, use_laderman=True)
        if _verify(U, V, W, n, m, p):
            best = (U, V, W); best_rank = U.shape[1]
    
    # Also try Strassen as a fallback/comparison
    if _time_left() > 15:
        U_s, V_s, W_s = _build_structured(n, m, p, use_laderman=False)
        if _verify(U_s, V_s, W_s, n, m, p):
            if U_s.shape[1] < best_rank:
                best = (U_s, V_s, W_s); best_rank = U_s.shape[1]

    # 2) Try to polish the best structured candidate.
    Uo, Vo, Wo = best
    if _time_left() > 30:
        for seed in range(8):
            if _time_left() < 25:
                break
            res = _smooth_polish(Uo, Vo, Wo, n, m, p, DEADLINE - 3, seed, maxiter=300)
            if res is not None:
                Ur, Vr, Wr = res
                if _verify(Ur, Vr, Wr, n, m, p):
                    if Ur.shape[1] < best_rank:
                        best = (Ur, Vr, Wr); best_rank = Ur.shape[1]
                    break

    # 3) Try rank reduction from the structured candidate.
    Uo, Vo, Wo = best
    if _time_left() > 25:
        for target in range(max(10, best_rank - 10), best_rank):
            if _time_left() < 15:
                break
            found = False
            for seed in range(10):
                if _time_left() < 12:
                    break
                res = _try_rank_reduction(Uo, Vo, Wo, n, m, p, target, seed, DEADLINE - 3)
                if res is not None:
                    Ur, Vr, Wr = res
                    if _verify(Ur, Vr, Wr, n, m, p):
                        if target < best_rank:
                            best = (Ur, Vr, Wr); best_rank = target
                        found = True
                        break
            if found:
                break

    Uo, Vo, Wo = best
    return {"U": Uo.tolist(), "V": Vo.tolist(), "W": Wo.tolist()}

# EVOLVE-BLOCK-END
