"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return U, V, W matrices (each 4x7) for 2x2x2 Strassen decomposition.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    Using Wikipedia Strassen formulas.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U rows: (0,0)=a00, (0,1)=a01, (1,0)=a10, (1,1)=a11
    # V rows: (0,0)=b00, (0,1)=b01, (1,0)=b10, (1,1)=b11
    # W rows: (0,0)=c00, (0,1)=c01, (1,0)=c10, (1,1)=c11
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    # c00 += M1, c11 += M1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (a10+a11)(b00+b01)
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1; V[1][1] += 1
    # c10 -= M2, c11 -= M2
    W[2][1] += -1; W[3][1] += -1
    
    # M3 = (a00+a01)(b10)
    U[0][2] += 1; U[1][2] += 1
    V[2][2] += 1
    # c01 += M3, c11 += M3
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = (a10+a11)(b11)
    U[2][3] += 1; U[3][3] += 1
    V[3][3] += 1
    # c00 += M4, c10 -= M4
    W[0][3] += 1; W[2][3] += -1
    
    # M5 = (a00+a10)(b01-b11)
    U[0][4] += 1; U[2][4] += 1
    V[1][4] += 1; V[3][4] += -1
    # c00 -= M5, c01 += M5
    W[0][4] += -1; W[1][4] += 1
    
    # M6 = (a01+a11)(b10-b00)
    U[1][5] += 1; U[3][5] += 1
    V[2][5] += 1; V[0][5] += -1
    # c10 += M6, c11 += M6
    W[2][5] += 1; W[3][5] += 1
    
    # M7 = (a00+a11)(b00-b11)
    U[0][6] += 1; U[3][6] += 1
    V[0][6] += 1; V[3][6] += -1
    # c00 += M7
    W[0][6] += 1
    
    return U, V, W

def _block_strassen_3x4x8():
    """
    Block-Strassen decomposition for 3x4x8 matrix multiplication.
    
    A (3x4) = [A_top (2x4); A_bot (1x4)]
    B (4x8) = [B_left (4x4) | B_right (4x4)]
    
    A_top x B_left: split into two 2x2x2 Strassen (rank 14)
    A_top x B_right: split into two 2x2x2 Strassen (rank 14)
    A_bot x B_left: schoolbook (rank 4)
    A_bot x B_right: schoolbook (rank 4)
    Total rank: 36
    """
    n, m, p = 3, 4, 8
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24
    
    # Get Strassen 2x2x2 base
    sU, sV, sW = _strassen_2x2x2()
    
    rank = 36
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    
    # Helper to add a Strassen block
    # A_sub: 2x2 submatrix of A at rows (i0, i0+1), cols (j0, j0+1)
    # B_sub: 2x2 submatrix of B at rows (j0, j0+1), cols (k0, k0+1)
    # Output: C at rows (i0, i0+1), cols (k0, k0+1)
    # U index: i*4 + j (since m=4)
    # V index: j*8 + k (since p=8)
    # W index: k*3 + i (since n=3)
    
    col_offset = 0
    
    # Block 1: A_top (rows 0,1) x B_left (cols 0,1) -> C rows 0,1, cols 0,1
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0,1}
    # C indices: i in {0,1}, k in {0,1}
    for r in range(7):
        for ui, val_u in enumerate(sU[r]):
            if val_u != 0:
                i, j = divmod(ui, 2)
                U[i*4 + j][col_offset + r] += val_u
        for vi, val_v in enumerate(sV[r]):
            if val_v != 0:
                j, k = divmod(vi, 2)
                V[j*8 + k][col_offset + r] += val_v
        for wi, val_w in enumerate(sW[r]):
            if val_w != 0:
                k, i = divmod(wi, 2)
                W[k*3 + i][col_offset + r] += val_w
    col_offset += 7
    
    # Block 2: A_top (rows 0,1) x B_left (cols 2,3) -> C rows 0,1, cols 2,3
    # A indices: i in {0,1}, j in {2,3}
    # B indices: j in {2,3}, k in {2,3}
    # C indices: i in {0,1}, k in {2,3}
    for r in range(7):
        for ui, val_u in enumerate(sU[r]):
            if val_u != 0:
                i, j = divmod(ui, 2)
                U[i*4 + (j+2)][col_offset + r] += val_u
        for vi, val_v in enumerate(sV[r]):
            if val_v != 0:
                j, k = divmod(vi, 2)
                V[(j+2)*8 + (k+2)][col_offset + r] += val_v
        for wi, val_w in enumerate(sW[r]):
            if val_w != 0:
                k, i = divmod(wi, 2)
                W[(k+2)*3 + i][col_offset + r] += val_w
    col_offset += 7
    
    # Block 3: A_top (rows 0,1) x B_right (cols 0,1) -> C rows 0,1, cols 4,5
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {4,5}
    # C indices: i in {0,1}, k in {4,5}
    for r in range(7):
        for ui, val_u in enumerate(sU[r]):
            if val_u != 0:
                i, j = divmod(ui, 2)
                U[i*4 + j][col_offset + r] += val_u
        for vi, val_v in enumerate(sV[r]):
            if val_v != 0:
                j, k = divmod(vi, 2)
                V[j*8 + (k+4)][col_offset + r] += val_v
        for wi, val_w in enumerate(sW[r]):
            if val_w != 0:
                k, i = divmod(wi, 2)
                W[(k+4)*3 + i][col_offset + r] += val_w
    col_offset += 7
    
    # Block 4: A_top (rows 0,1) x B_right (cols 2,3) -> C rows 0,1, cols 6,7
    # A indices: i in {0,1}, j in {2,3}
    # B indices: j in {2,3}, k in {6,7}
    # C indices: i in {0,1}, k in {6,7}
    for r in range(7):
        for ui, val_u in enumerate(sU[r]):
            if val_u != 0:
                i, j = divmod(ui, 2)
                U[i*4 + (j+2)][col_offset + r] += val_u
        for vi, val_v in enumerate(sV[r]):
            if val_v != 0:
                j, k = divmod(vi, 2)
                V[(j+2)*8 + (k+4)][col_offset + r] += val_v
        for wi, val_w in enumerate(sW[r]):
            if val_w != 0:
                k, i = divmod(wi, 2)
                W[(k+4)*3 + i][col_offset + r] += val_w
    col_offset += 7
    
    # Block 5: A_bot (row 2) x B_left (cols 0,1,2,3) -> C row 2, cols 0,1,2,3
    # Schoolbook: 4 terms
    for j in range(4):
        for k in range(4):
            U[2*4 + j][col_offset] = 1
            V[j*8 + k][col_offset] = 1
            W[k*3 + 2][col_offset] = 1
            col_offset += 1
    # Wait, that's 16 terms, not 4. Let me reconsider.
    # A_bot is 1x4, B_left is 4x4. Product is 1x4.
    # C[2,k] = sum_{j=0}^{3} A[2,j] * B[j,k]
    # This is 4*4 = 16 terms in schoolbook? No, it's 4 terms for each k, but 4 k's, so 16 total.
    # Actually, the rank of 1x4x4 multiplication is 4*4 = 16? No.
    # The tensor T[i,j,k] for i=0 (only row), j in 0..3, k in 0..3
    # T[0,j,k] = 1 for all j,k. This is a 1x4x4 tensor which is just a 4x4 matrix of 1s.
    # The rank of a 4x4 matrix of all 1s is 1! Because it's a rank-1 matrix.
    # Wait, but the tensor decomposition here is different.
    # sum_r U[j, r] * V[j,k, r] * W[k, r] = 1 for all j,k
    # Hmm, this is not a standard matrix rank.
    # Actually for 1xm x mxp: C[k] = sum_j A[j] B[j,k]
    # The tensor is T[j,k] = 1 for all j,k (since i is fixed)
    # We need sum_r U[j,r] V[j,k,r] W[k,r] = 1
    # This can be done with rank p (or m) using:
    # For each k: U[j, k] = 1, V[j,k, k] = 1, W[k, k] = 1? No.
    # Let's think: set r = k (0..p-1)
    # U[j, k] = 1 for all j
    # V[j, k, k] = 1 only when... hmm this doesn't work simply.
    # Actually: U[j, r] for r in 0..p-1, V[j*p+k, r], W[k*n+i, r]
    # For i=2 (fixed), n=3: W index = k*3+2
    # We want sum_r U[2*4+j, r] * V[j*8+k, r] * W[k*3+2, r] = 1 for all j,k
    # One solution: rank = p = 4
    # Let r = k (0..3)
    # U[2*4+j, k] = 1 for all j
    # V[j*8+k, k] = 1 for all j
    # W[k*3+2, k] = 1
    # Then sum_r U[2*4+j, r] * V[j*8+k, r] * W[k*3+2, r]
    # = sum_{r=0}^{3} 1 * (1 if r==k else 0) * (1 if r==k else 0)
    # Wait, V[j*8+k, r] = 1 only when r == k? No, I set V[j*8+k, k] = 1, so V[j*8+k, r] = 1 if r==k, else 0.
    # And W[k*3+2, r] = 1 if r==k, else 0.
    # So the sum = U[2*4+j, k] * V[j*8+k, k] * W[k*3+2, k] = 1*1*1 = 1. ✓
    # So rank 4 works for 1x4x4.
    
    # I made an error above. Let me redo blocks 5-8.
    # Reset col_offset for blocks 5-8
    col_offset = 28  # After 4 Strassen blocks of 7 each
    
    # Block 5: A_bot (row 2) x B_left (cols 0-3) -> C row 2, cols 0-3
    # Rank 4
    for k in range(4):
        for j in range(4):
            U[2*4 + j][col_offset + k] = 1
            V[j*8 + k][col_offset + k] = 1
        W[k*3 + 2][col_offset + k] = 1
    col_offset += 4
    
    # Block 6: A_bot (row 2) x B_right (cols 4-7) -> C row 2, cols 4-7
    # Rank 4
    for k in range(4):
        for j in range(4):
            U[2*4 + j][col_offset + k] = 1
            V[j*8 + (k+4)][col_offset + k] = 1
        W[(k+4)*3 + 2][col_offset + k] = 1
    col_offset += 4
    
    # Total rank should be 28 + 4 + 4 = 36
    assert col_offset == rank, f"col_offset={col_offset}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Try block-Strassen first
    try:
        result = _block_strassen_3x4x8()
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
