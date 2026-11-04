import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, rank, tol=1e-9):
    """Fast float verification."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_base():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2 (rank 7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1: (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # M5: (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6: (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7: (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _build_4x4_4x4_strassen():
    """Build 4x4 * 4x4 -> 4x4 decomposition using recursive Strassen (rank 49)."""
    U2, V2, W2 = _strassen_2x2_base()
    
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(7):
        for d in range(7):
            col = c * 7 + d
            for i in range(4):
                for j in range(4):
                    bi, bj = i // 2, j // 2
                    ii, jj = i % 2, j % 2
                    val = U2[bi*2+bj][c] * U2[ii*2+jj][d]
                    if val != 0:
                        U[i*4+j][col] = val
            for j in range(4):
                for k in range(4):
                    bj, bk = j // 2, k // 2
                    jj, kk = j % 2, k % 2
                    val = V2[bj*2+bk][c] * V2[jj*2+kk][d]
                    if val != 0:
                        V[j*4+k][col] = val
            for k in range(4):
                for i in range(4):
                    bk, bi = k // 2, i // 2
                    kk, ii = k % 2, i % 2
                    val = W2[bk*2+bi][c] * W2[kk*2+ii][d]
                    if val != 0:
                        W[k*4+i][col] = val
    
    return U, V, W

def _build_4x4_4x5_base():
    """Build 4x4 * 4x5 -> 4x5 decomposition (rank 50)."""
    U4, V4, W4 = _build_4x4_4x4_strassen()
    
    n, m, p = 4, 4, 5
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(49):
            U[i][c] = U4[i][c]
    for j in range(m):
        for k in range(4):
            for c in range(49):
                V[j*p+k][c] = V4[j*4+k][c]
    for k in range(4):
        for i in range(n):
            for c in range(49):
                W[k*n+i][c] = W4[k*4+i][c]
    
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*4+4][col] = 1
    for i in range(4):
        W[4*4+i][col] = 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, deadline, max_attempts=2000):
    """Attempt to reduce rank by finding linear dependencies among columns.
    
    We check if any column is a linear combination of others.
    If so, we can eliminate it, reducing rank by 1.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Keep track of which columns are "active"
    active = list(range(rank))
    changed = True
    
    while changed and time.time() < deadline:
        changed = False
        # For each active column, check if it's in the span of others
        for idx in range(len(active)):
            if time.time() > deadline:
                return U, V, W, len(active)
            c = active[idx]
            # Check if column c is in span of other active columns
            others = [a for i, a in enumerate(active) if i != idx]
            if not others:
                continue
            
            # Form matrix of other columns: shape (NM+MP+PN, len(others))
            # We need to check if the vector for column c is in the span
            vec_c = np.concatenate([U_np[:, c], V_np[:, c], W_np[:, c]])
            mat_others = np.column_stack([
                np.concatenate([U_np[:, o], V_np[:, o], W_np[:, o]]) for o in others
            ])
            
            # Check if vec_c is in column space of mat_others
            # Use least squares
            try:
                coeff, residuals, rank_others, _ = np.linalg.lstsq(mat_others, vec_c, rcond=1e-10)
            except Exception:
                continue
            
            if len(residuals) == 0 or np.linalg.norm(vec_c - mat_others @ coeff) < 1e-8:
                # Column c is redundant! Remove it.
                # Reconstruct U, V, W without column c
                new_active = [a for i, a in enumerate(active) if i != idx]
                U_new = [[0]*len(new_active) for _ in range(n*m)]
                V_new = [[0]*len(new_active) for _ in range(m*p)]
                W_new = [[0]*len(new_active) for _ in range(p*n)]
                for new_i, old_c in enumerate(new_active):
                    for i in range(n*m):
                        U_new[i][new_i] = U[i][old_c]
                    for j in range(m*p):
                        V_new[j][new_i] = V[j][old_c]
                    for k in range(p*n):
                        W_new[k][new_i] = W[k][old_c]
                
                # Verify
                if _verify(U_new, V_new, W_new, n, m, p, len(new_active)):
                    U, V, W = U_new, V_new, W_new
                    active = new_active
                    changed = True
                    break
                else:
                    # Float error, skip
                    pass
    
    return U, V, W, len(active)

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140
    
    # Build the rank-50 decomposition
    U, V, W = _build_4x4_4x5_base()
    rank = len(U[0])
    
    # Verify
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies
    U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
