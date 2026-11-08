"""Structural change: Initialize with schoolbook decomposition and prune columns.
Use alternating linear least squares to adjust the remaining columns to maintain
validity. This exploits the near-linear dependency of the schoolbook tensor terms."""
import time
import random

try:
    import numpy as np
except Exception:
    np = None

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_numpy(U, V, W, n, m, p):
    if np is None:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            target = np.eye(m, dtype=np.float64)
            if not np.allclose(M, target, atol=1e-9):
                return False
    return True

def _solve_for_U(V, W, n, m, p, R):
    U = [[0] * R for _ in range(n * m)]
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, R), dtype=np.float64)
            for j in range(m):
                for c in range(R):
                    A[j, c] = W[k * n + i][c] * V[j * p + k][c]
            # Solve U_block @ A.T = I_m
            # U_block is m x R
            # (U_block @ A.T)[j, j'] = sum_c U_block[j,c] * A[j',c] = delta_{j,j'}
            # So U_block @ A.T = I_m
            # U_block = pinv(A.T)
            try:
                U_block = np.linalg.pinv(A.T)
            except Exception:
                U_block = np.zeros((m, R), dtype=np.float64)
            for j in range(m):
                for c in range(R):
                    U[i * m + j][c] = int(round(U_block[j, c]))
    return U

def _solve_for_V(U, W, n, m, p, R):
    V = [[0] * R for _ in range(m * p)]
    for k in range(p):
        for i in range(n):
            B = np.zeros((m, R), dtype=np.float64)
            for j in range(m):
                for c in range(R):
                    B[j, c] = W[k * n + i][c] * U[i * m + j][c]
            # B @ V_block.T = I_m
            # V_block is m x R
            # (B @ V_block.T)[j, j'] = sum_c B[j,c] * V_block[j',c] = delta_{j,j'}
            # V_block.T = pinv(B)
            # V_block = pinv(B).T
            try:
                V_block = np.linalg.pinv(B).T
            except Exception:
                V_block = np.zeros((m, R), dtype=np.float64)
            for jp in range(m):
                for c in range(R):
                    V[jp * p + k][c] = int(round(V_block[jp, c]))
    return V

def _solve_for_W(U, V, n, m, p, R):
    W = [[0] * R for _ in range(p * n)]
    for i in range(n):
        for k in range(p):
            A = np.zeros((m * m, R), dtype=np.float64)
            b = np.zeros(m * m, dtype=np.float64)
            idx = 0
            for j in range(m):
                for jp in range(m):
                    for c in range(R):
                        A[idx, c] = U[i * m + j][c] * V[jp * p + k][c]
                    b[idx] = 1.0 if j == jp else 0.0
                    idx += 1
            try:
                W_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                W_row = np.zeros(R, dtype=np.float64)
            for c in range(R):
                W[k * n + i][c] = int(round(W_row[c]))
    return W

def _try_rank_from_schoolbook(R, n, m, p, deadline, rng):
    # Start with schoolbook, keep first R columns
    U0, V0, W0 = _build_schoolbook(n, m, p)
    U = [row[:R] for row in U0]
    V = [row[:R] for row in V0]
    W = [row[:R] for row in W0]
    
    # Iterate ALS
    for it in range(20):
        if time.time() > deadline - 20.0:
            return None
        U = _solve_for_U(V, W, n, m, p, R)
        V = _solve_for_V(U, W, n, m, p, R)
        W = _solve_for_W(U, V, n, m, p, R)
        
        if _verify_numpy(U, V, W, n, m, p):
            # Check if integers
            U_int = [[int(round(x)) for x in row] for row in U]
            V_int = [[int(round(x)) for x in row] for row in V]
            W_int = [[int(round(x)) for x in row] for row in W]
            if _verify_numpy(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    
    # Try to reduce rank from 120 downwards
    # Schoolbook is 120. 
    # We know rank is likely lower, but we don't know the exact minimal.
    # Try a few steps down.
    for R in range(120, 100, -1):
        if time.time() > deadline - 30.0:
            break
        result = _try_rank_from_schoolbook(R, n, m, p, deadline, rng)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
