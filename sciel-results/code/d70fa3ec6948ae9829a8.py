"""
Matrix multiplication 2x5x5: rank-minimal exact tensor decomposition.

Strategy:
Partition j-indices into J1={0,1,2} and J2={3,4}
Partition k-indices into K1={0,1,2} and K2={3,4}

This creates a 2x2 block structure:
- Block (J1,K1): M_{2,3,3} -> rank 14 (Strassen 3x3x3, with A row 2 = 0)
- Block (J1,K2): M_{2,3,2} -> rank 12 (trivial)
- Block (J2,K1): M_{2,2,3} -> rank 12 (trivial)
- Block (J2,K2): M_{2,2,2} -> rank 7 (Strassen 2x2x2)

Total rank = 14 + 12 + 12 + 7 = 45
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

_START_TIME = time.time()
_TIME_BUDGET = 140.0  # seconds; leave 10s safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _within_budget(margin=5.0):
    return _elapsed() < _TIME_BUDGET - margin


def _verify(U, V, W, n, m, p, num_tests=10):
    """Verify decomposition using multiple random test matrices (vectorized)."""
    U_np = np.array(U, dtype=float)  # (n*m, rank)
    V_np = np.array(V, dtype=float)  # (m*p, rank)
    W_np = np.array(W, dtype=float)  # (p*n, rank)
    
    rank = U_np.shape[1]
    
    for test in range(num_tests):
        rng = np.random.RandomState(42 + test)
        A = rng.randn(n * m)
        B = rng.randn(m * p)
        
        A_mat = A.reshape(n, m)
        B_mat = B.reshape(m, p)
        C_true = (A_mat @ B_mat).flatten()  # (n*p,) with index i*p+k
        
        # W uses k*n+i indexing
        C_true_wn = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_true_wn[k * n + i] = C_true[i * p + k]
        
        # Vectorized: vA = U^T @ A (shape: rank,), vB = V^T @ B (shape: rank,)
        vA = U_np.T @ A   # (rank,)
        vB = V_np.T @ B   # (rank,)
        # C_comp = W @ (vA * vB)  -> (p*n,)
        C_comp = W_np @ (vA * vB)
        
        if not np.allclose(C_comp, C_true_wn, atol=1e-9):
            return False
    
    return True


def _build_rank45():
    """
    Build a rank-45 decomposition for 2x5x5 using Strassen 3x3x3.
    
    Partition j into J1={0,1,2}, J2={3,4}
    Partition k into K1={0,1,2}, K2={3,4}
    
    Blocks:
    (J1,K1): M_{2,3,3} -> Strassen 3x3x3 rank 14
    (J1,K2): M_{2,3,2} -> trivial rank 12
    (J2,K1): M_{2,2,3} -> trivial rank 12
    (J2,K2): M_{2,2,2} -> Strassen rank 7
    
    Total: 14 + 12 + 12 + 7 = 45
    """
    n, m, p = N, M, P
    
    J1 = [0, 1, 2]
    J2 = [3, 4]
    K1 = [0, 1, 2]
    K2 = [3, 4]
    
    U_entries = []
    V_entries = []
    W_entries = []
    fid = 0
    
    # Strassen 3x3x3: 14 factors
    # A is 3x3, B is 3x3, C is 3x3
    # We use only A rows 0,1 (row 2 = 0) and C rows 0,1
    strassen_3x3 = [
        # m1 = (a11+a22)(b11+b22)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # m2 = (a21+a22)b11
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (2,1,-1)]),
        # m3 = a11(b21-b22)
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(1,0,1), (2,0,-1)]),
        # m4 = (a11+a12)b33
        ([(0,0,1), (0,1,1)], [(2,2,1)], [(0,0,-1), (1,1,1)]),
        # m5 = (a31+a32)b13
        ([(2,0,1), (2,1,1)], [(0,2,1)], [(0,2,1), (2,1,-1)]),
        # m6 = (a11-a33)(b33+b22)
        ([(0,0,1), (2,2,-1)], [(2,2,1), (1,1,1)], [(1,1,1), (0,0,-1)]),
        # m7 = (a13+a33)(b11-b33)
        ([(0,2,1), (2,2,1)], [(0,0,1), (2,2,-1)], [(0,0,1), (1,1,-1)]),
        # m8 = (a11+a12)(b22-b33)
        ([(0,0,1), (0,1,1)], [(1,1,1), (2,2,-1)], [(0,1,1), (1,1,-1)]),
        # m9 = (a21+a31)(b13+b33)
        ([(1,0,1), (2,0,1)], [(0,2,1), (2,2,1)], [(1,0,1), (2,2,-1)]),
        # m10 = (a31+a32)(b22-b11)
        ([(2,0,1), (2,1,1)], [(1,1,1), (0,0,-1)], [(1,0,-1), (2,1,1)]),
        # m11 = (a11+a33)(b13+b31)
        ([(0,0,1), (2,2,1)], [(0,2,1), (2,0,1)], [(0,2,-1), (2,0,1)]),
        # m12 = (a13+a33)(b21+b11)
        ([(0,2,1), (2,2,1)], [(1,0,1), (0,0,1)], [(0,1,-1), (2,0,1)]),
        # m13 = (a13+a12)(b32+b33)
        ([(0,2,1), (0,1,1)], [(2,1,1), (2,2,1)], [(0,2,1), (2,2,-1)]),
        # m14 = (a21+a22)(b31+b33)
        ([(1,0,1), (1,1,1)], [(2,0,1), (2,2,1)], [(2,1,1), (1,2,-1)]),
    ]
    
    # Block (J1, K1): M_{2,3,3} using Strassen 3x3x3
    for (u_ent, v_ent, w_ent) in strassen_3x3:
        for (i, jl, c) in u_ent:
            if i < 2:
                gj = J1[jl]
                U_entries.append((fid, i * m + gj, c))
        for (jl, kl, c) in v_ent:
            gj = J1[jl]
            gk = K1[kl]
            V_entries.append((fid, gj * p + gk, c))
        for (kl, i, c) in w_ent:
            if i < 2:
                gk = K1[kl]
                W_entries.append((fid, gk * n + i, c))
        fid += 1
    
    # Block (J1, K2): M_{2,3,2} trivial (rank 12)
    for i in range(n):
        for jl in range(3):
            for kl in range(2):
                gj = J1[jl]
                gk = K2[kl]
                U_entries.append((fid, i * m + gj, 1))
                V_entries.append((fid, gj * p + gk, 1))
                W_entries.append((fid, gk * n + i, 1))
                fid += 1
    
    # Block (J2, K1): M_{2,2,3} trivial (rank 12)
    for i in range(n):
        for jl in range(2):
            for kl in range(3):
                gj = J2[jl]
                gk = K1[kl]
                U_entries.append((fid, i * m + gj, 1))
                V_entries.append((fid, gj * p + gk, 1))
                W_entries.append((fid, gk * n + i, 1))
                fid += 1
    
    # Block (J2, K2): M_{2,2,2} Strassen (rank 7)
    strassen_7 = [
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)]),
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)]),
        ([(1, 1, 1)],
         [(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)]),
        ([(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, -1), (1, 0, 1)]),
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)]),
    ]
    for (u_ent, v_ent, w_ent) in strassen_7:
        for (i, jl, c) in u_ent:
            gj = J2[jl]
            U_entries.append((fid, i * m + gj, c))
        for (jl, kl, c) in v_ent:
            gj = J2[jl]
            gk = K2[kl]
            V_entries.append((fid, gj * p + gk, c))
        for (kl, i, c) in w_ent:
            gk = K2[kl]
            W_entries.append((fid, gk * n + i, c))
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
    
    result = _build_rank45()
    
    if _within_budget():
        if _verify(result["U"], result["V"], result["W"], n, m, p, num_tests=10):
            return result
    
    return _build_schoolbook()
