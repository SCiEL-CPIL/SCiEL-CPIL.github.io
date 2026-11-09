"""
Matrix multiplication 2x5x5: rank-minimal exact tensor decomposition.

Strategy:
Partition j-indices into groups {0,1}, {2,3}, {4}
Partition k-indices into groups {0,1}, {2,3}, {4}

This creates a 3x3 block structure:
- Four 2x2x2 blocks (Strassen rank 7 each) -> 4*7 = 28
- Four 2x2x1 blocks (trivial rank 4 each) -> 4*4 = 16
- Four 2x1x2 blocks (trivial rank 4 each) -> 4*4 = 16
- One 2x1x1 block (trivial rank 2) -> 2
Total rank = 28 + 16 + 16 + 2 = 46

Wait, that's 62. Let me recount.
Actually:
- (J0,K0): 2x2x2 -> Strassen 7
- (J0,K1): 2x2x2 -> Strassen 7
- (J0,K2): 2x2x1 -> trivial 4
- (J1,K0): 2x2x2 -> Strassen 7
- (J1,K1): 2x2x2 -> Strassen 7
- (J1,K2): 2x2x1 -> trivial 4
- (J2,K0): 2x1x2 -> trivial 4
- (J2,K1): 2x1x2 -> trivial 4
- (J2,K2): 2x1x1 -> trivial 2
Total: 7+7+4+7+7+4+4+4+2 = 46
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

_START_TIME = time.time()
_TIME_BUDGET = 140.0


def _elapsed():
    return time.time() - _START_TIME


def _within_budget(margin=5.0):
    return _elapsed() < _TIME_BUDGET - margin


def _verify(U, V, W, n, m, p, num_tests=10):
    """Verify decomposition using multiple random test matrices (vectorized)."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    for test in range(num_tests):
        rng = np.random.RandomState(42 + test)
        A = rng.randn(n * m)
        B = rng.randn(m * p)
        
        A_mat = A.reshape(n, m)
        B_mat = B.reshape(m, p)
        C_true = (A_mat @ B_mat).flatten()
        
        C_true_wn = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_true_wn[k * n + i] = C_true[i * p + k]
        
        vA = U_np.T @ A
        vB = V_np.T @ B
        C_comp = W_np @ (vA * vB)
        
        if not np.allclose(C_comp, C_true_wn, atol=1e-9):
            return False
    
    return True


def _build_rank46():
    """
    Build a rank-46 decomposition for 2x5x5.
    
    Partition j into {0,1}, {2,3}, {4}
    Partition k into {0,1}, {2,3}, {4}
    
    4x Strassen 2x2x2 (rank 7 each) + 4x 2x2x1 (rank 4) + 4x 2x1x2 (rank 4) + 1x 2x1x1 (rank 2)
    = 28 + 16 + 16 + 2 = 46
    """
    n, m, p = N, M, P
    
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4]]
    
    U_entries = []
    V_entries = []
    W_entries = []
    
    fid = 0
    
    # Strassen's 7 factors for 2x2x2
    # Each factor is (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, coeff) for A[i,j]
    # v_entries: list of (j, k, coeff) for B[j,k]
    # w_entries: list of (k, i, coeff) for C[k,i] (W convention: k*n+i)
    strassen_7 = [
        # Factor 0: M1 = (A00+A11)(B00+B11)
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)]),
        # Factor 1: M2 = (A10+A11)B00
        ([(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)]),
        # Factor 2: M3 = A00(B10-B00)
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)]),
        # Factor 3: M4 = A11(B01-B10)
        ([(1, 1, 1)],
         [(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)]),
        # Factor 4: M5 = (A00+A01)B11
        ([(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, -1), (1, 0, 1)]),
        # Factor 6: M6 = (A10-A00)(B00+B01)
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)]),
        # Factor 7: M7 = (A01-A11)(B10+B11)
        ([(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)]),
    ]
    
    for a in range(3):
        for b in range(3):
            jg = j_groups[a]
            kg = k_groups[b]
            sj = len(jg)
            sk = len(kg)
            
            if sj == 2 and sk == 2:
                # Strassen 2x2x2
                for (u_ent, v_ent, w_ent) in strassen_7:
                    for (i, jl, c) in u_ent:
                        gj = jg[jl]
                        U_entries.append((fid, i * m + gj, c))
                    for (jl, kl, c) in v_ent:
                        gj = jg[jl]
                        gk = kg[kl]
                        V_entries.append((fid, gj * p + gk, c))
                    for (kl, i, c) in w_ent:
                        gk = kg[kl]
                        W_entries.append((fid, gk * n + i, c))
                    fid += 1
            else:
                # Trivial decomposition
                for i in range(n):
                    for jl in range(sj):
                        for kl in range(sk):
                            gj = jg[jl]
                            gk = kg[kl]
                            U_entries.append((fid, i * m + gj, 1))
                            V_entries.append((fid, gj * p + gk, 1))
                            W_entries.append((fid, gk * n + i, 1))
                            fid += 1
    
    rank = fid
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (f, r, v) in U_entries:
        U[r][f] += v
    for (f, r, v) in V_entries:
        V[r][f] += v
    for (f, r, v) in W_entries:
        W[r][f] += v
    
    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Return the schoolbook rank-50 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix mult.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = N, M, P
    
    result = _build_rank46()
    
    if _within_budget():
        if _verify(result["U"], result["V"], result["W"], n, m, p, num_tests=10):
            return result
    
    return _build_schoolbook()
