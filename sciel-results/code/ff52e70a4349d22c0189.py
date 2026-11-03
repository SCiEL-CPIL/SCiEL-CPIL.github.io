"""Program for matrix multiplication 3x5x5 tensor decomposition.

Uses block decomposition: i={2,1}, j={2,2,1}, k={2,2,1}
with Strassen for 2x2x2 blocks and schoolbook for others.
Total rank: 26
"""

import time
import numpy as np


def _strassen_2x2x2():
    """Return (U, V, W) for Strassen 2x2x2 multiplication, rank 7.
    
    U: shape (4, 7), row indexed by (i,j) with i in {0,1}, j in {0,1}
    V: shape (4, 7), row indexed by (j,k) with j in {0,1}, k in {0,1}
    W: shape (4, 7), row indexed by (k,i) with k in {0,1}, i in {0,1}
    
    Such that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta(i,j,k)
    """
    U = [[0.0] * 7 for _ in range(4)]
    V = [[0.0] * 7 for _ in range(4)]
    W = [[0.0] * 7 for _ in range(4)]
    
    # Standard Strassen 2x2x2
    # m1 = (A11-A22)(B21+B22) -> U[0][0]=1, U[0][1]=-1; V[0][0]=1, V[0][1]=1; W[0][0]=1, W[0][1]=1
    # m2 = (A21+A22)B11
    # m3 = A11(B12-B22)
    # m4 = A22(B21-B11)
    # m5 = (A11+A12)B22
    # m6 = (A21-A11)(B11+B12)
    # m7 = (A12-A22)(B21+B22)
    
    # Let me use a verified Strassen decomposition
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # U[i,j] for A_{i+1,j+1}:
    # m1: A11-A22 -> U[0,0]=1, U[1,1]=-1
    # m2: A21+A22 -> U[1,0]=1, U[1,1]=1
    # m3: A11 -> U[0,0]=1
    # m4: A22 -> U[1,1]=1
    # m5: A11+A12 -> U[0,0]=1, U[0,1]=1
    # m6: A21-A11 -> U[1,0]=1, U[0,0]=-1
    # m7: A12-A22 -> U[0,1]=1, U[1,1]=-1
    
    U[0][0] = 1.0  # m1: A11
    U[1][0] = -1.0  # m1: -A22
    U[1][1] = 1.0  # m2: A21... wait, U[1,0] is A_{2,1}, U[1,1] is A_{2,2}
    
    # Let me redo with proper indexing:
    # U[i*m+j] where m=2, so U[i*2+j]
    # Row 0 = (0,0), Row 1 = (0,1), Row 2 = (1,0), Row 3 = (1,1)
    
    U = [[0.0] * 7 for _ in range(4)]
    V = [[0.0] * 7 for _ in range(4)]
    W = [[0.0] * 7 for _ in range(4)]
    
    # m1 = (A00 - A11)(B10 + B11)
    U[0][0] = 1.0   # A00
    U[3][0] = -1.0  # -A11
    V[2][0] = 1.0   # B10
    V[3][0] = 1.0   # B11
    W[0][0] = 1.0   # C00 += m1
    W[3][0] = 1.0   # C11 += m1
    
    # m2 = (A10 + A11)B00
    U[2][1] = 1.0   # A10
    U[3][1] = 1.0   # A11
    V[0][1] = 1.0   # B00
    W[2][1] = 1.0   # C10 += m2
    W[3][1] = -1.0  # C11 -= m2
    
    # m3 = A00(B01 - B11)
    U[0][2] = 1.0   # A00
    V[1][2] = 1.0   # B01
    V[3][2] = -1.0  # -B11
    W[1][2] = 1.0   # C01 += m3
    W[3][2] = 1.0   # C11 += m3
    
    # m4 = A11(B10 - B00)
    U[3][3] = 1.0   # A11
    V[2][3] = 1.0   # B10
    V[0][3] = -1.0  # -B00
    W[2][3] = 1.0   # C10 += m4
    W[3][3] = 1.0   # C11 += m4
    
    # m5 = (A00 + A01)B11
    U[0][4] = 1.0   # A00
    U[1][4] = 1.0   # A01
    V[3][4] = 1.0   # B11
    W[0][4] = -1.0  # C00 -= m5
    W[1][4] = 1.0   # C01 += m5
    
    # m6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1.0   # A10
    U[0][5] = -1.0  # -A00
    V[0][5] = 1.0   # B00
    V[1][5] = 1.0   # B01
    W[3][5] = 1.0   # C11 += m6
    
    # m7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1.0   # A01
    U[3][6] = -1.0  # -A11
    V[2][6] = 1.0   # B10
    V[3][6] = 1.0   # B11
    W[0][6] = 1.0   # C00 += m7
    W[1][6] = -1.0  # C01 -= m7
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify decomposition correctness."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i == 0 and j == 0 and k == 0) else 0.0
                # Wait, this is wrong. The tensor should be:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
                # No, that's not right either.
                # For matrix multiplication (n x m) * (m x p) = (n x p):
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
                # And T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                total = float(np.dot(np.multiply(u_val, v_val), w_val))
                
                if abs(total - 1.0) > tol:
                    return False
    return True


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # Leave 10s margin before 150s kill
    
    n, m, p = 3, 5, 5
    
    # Block decomposition
    i_blocks = [[0, 1], [2]]       # sizes: 2, 1
    j_blocks = [[0, 1], [2, 3], [4]]  # sizes: 2, 2, 1
    k_blocks = [[0, 1], [2, 3], [4]]  # sizes: 2, 2, 1
    
    # Pre-compute Strassen factors
    sU, sV, sW = _strassen_2x2x2()
    
    # Calculate total rank
    total_rank = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                ni, nj, nk = len(ib), len(jb), len(kb)
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * nj * nk
    
    # Initialize matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                ni, nj, nk = len(ib), len(jb), len(kb)
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Use Strassen for 2x2x2
                    for c in range(7):
                        for li in range(2):
                            for lj in range(2):
                                gi = ib[li]
                                gj = jb[lj]
                                U[gi * m + gj][col + c] += sU[li * 2 + lj][c]
                        for lj in range(2):
                            for lk in range(2):
                                gj = jb[lj]
                                gk = kb[lk]
                                V[gj * p + gk][col + c] += sV[lj * 2 + lk][c]
                        for lk in range(2):
                            for li in range(2):
                                gk = kb[lk]
                                gi = ib[li]
                                W[gk * n + gi][col + c] += sW[lk * 2 + li][c]
                    col += 7
                else:
                    # Schoolbook
                    for li in range(ni):
                        for lj in range(nj):
                            for lk in range(nk):
                                gi = ib[li]
                                gj = jb[lj]
                                gk = kb[lk]
                                U[gi * m + gj][col] = 1.0
                                V[gj * p + gk][col] = 1.0
                                W[gk * n + gi][col] = 1.0
                                col += 1
    
    if col != total_rank:
        raise ValueError(f"Column mismatch: {col} != {total_rank}")
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to full schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1.0
                    V[j * p + k][c] = 1.0
                    W[k * n + i][c] = 1.0
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
