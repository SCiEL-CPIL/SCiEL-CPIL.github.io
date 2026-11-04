"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.

Uses a structured approach:
- Partition A's columns into groups to exploit structure
- For the 2x1 * 1x6 subproblem (A column 2 * B row 2), rank is 6 (not 1)
  because each output C[i,k] = A[i,2]*B[2,k] requires separate rank-1 terms
  for each (i,k) pair... wait, no. Let me reconsider.

Actually for 2x1 * 1x6: A is 2x1, B is 1x6, C is 2x6.
C[i,k] = A[i,0] * B[0,k]
This is a rank-2 tensor (it factors as a outer product structure).
In the tensor decomposition sense, we need rank = 2*1*6 = 12? No.
The tensor has entries T[i,j,k] = A[i,0]*B[0,k] where j is fixed at 0.
This is a rank-1 tensor in the (i,k) space for each fixed j... 
Actually the full tensor T[i,j,k] for the subproblem where j ranges over {0} only
has non-zero entries when j=0: T[i,0,k] = A[i,0]*B[0,k].
This can be written as sum over r of u_r[i]*v_r[k] where u_r = A[:,0] and v_r = B[0,:].
So it's rank 1 in the (i,k) bilinear form? No, it's rank 2 as a 3-tensor.

Let me just do the full schoolbook (rank 36) but also try a smarter partition.

For 2x3 * 3x6, one approach:
- Split into (2x2 * 2x6) + (2x1 * 1x6)
- 2x2 * 2x6: can use 3 Strassen blocks of 2x2*2x2, each rank 7 -> total 21
- 2x1 * 1x6: rank 6 (each of the 12 entries C[i,k] = A[i,2]*B[2,k], but these factor)

Wait, for 2x1 * 1x6: C[i,k] = A[i,2]*B[2,k]. The tensor T[i,j,k] is non-zero only at j=2.
T[i,2,k] = A[i,2]*B[2,k]. This is a rank-1 tensor (outer product of A[:,2] and B[2,:]).
So the subproblem has tensor rank 1!

Total rank: 21 + 1 = 22.

Let me verify: For 2x2 * 2x2 Strassen, rank is 7. For 2x2 * 2x6, we can view it as
three independent 2x2 * 2x2 blocks (columns 0-1, 2-3, 4-5 of B), each with rank 7.
Total for 2x2 * 2x6: 3*7 = 21.

For 2x1 * 1x6: A is 2x1 (column 2 of original A), B is 1x6 (row 2 of original B).
C[i,k] = A[i,2]*B[2,k]. The tensor entry T[i,j,k] = 1 if j=2, else 0, and when j=2,
T[i,2,k] = A[i,2]*B[2,k]. This is exactly the outer product of vector A[:,2] (length 2)
and vector B[2,:] (length 6), embedded in the 2x3x6 tensor at position j=2.
So rank 1 suffices.

Total: 21 + 1 = 22.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 6
    t_start = time.time()
    DEADLINE = 120.0  # seconds
    
    def verify(U, V, W, rank):
        """Verify the decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def build_strassen_2x2_2x2(U, V, W, col_offset, i0, j0, k0):
        """
        Add a Strassen 2x2*2x2 decomposition starting at column col_offset.
        A sub-block: rows i0..i0+1, cols j0..j0+1
        B sub-block: rows j0..j0+1, cols k0..k0+1
        C sub-block: rows i0..i0+1, cols k0..k0+1
        """
        # Strassen's 7 products for 2x2*2x2:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b11-b01)
        # M4 = a11*(b00-b10)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a01)*(b01+b11)
        # M7 = (a01-a10)*(b00+b10)
        #
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # Indices in global U, V, W:
        # U row for A[i,j]: i*m + j
        # V row for B[j,k]: j*p + k
        # W row for C[k,i]: k*n + i
        
        def urow(i, j):
            return (i0 + i) * m + (j0 + j)
        
        def vrow(j, k):
            return (j0 + j) * p + (k0 + k)
        
        def wrow(k, i):
            return (k0 + k) * n + (i0 + i)
        
        # M1 = (a00+a11)*(b00+b11)
        c = col_offset + 0
        U[urow(0,0), c] += 1.0
        U[urow(1,1), c] += 1.0
        V[vrow(0,0), c] += 1.0
        V[vrow(1,1), c] += 1.0
        W[wrow(0,0), c] += 1.0   # +M1 to c00
        W[wrow(1,1), c] += 1.0   # +M1 to c11
        
        # M2 = (a10+a11)*b00
        c = col_offset + 1
        U[urow(1,0), c] += 1.0
        U[urow(1,1), c] += 1.0
        V[vrow(0,0), c] += 1.0
        W[wrow(0,0), c] += 1.0   # +M2 to c00... wait
        # c00 = M1 + M4 - M5 + M7, c10 = M2 + M4
        # So M2 contributes to c10 only
        W[wrow(0,0), c] = 0.0  # undo
        W[wrow(0,1), c] += 1.0  # M2 to c10 (k=0, i=1)
        
        # M3 = a00*(b11-b01)
        c = col_offset + 2
        U[urow(0,0), c] += 1.0
        V[vrow(1,1), c] += 1.0
        V[vrow(0,1), c] -= 1.0
        W[wrow(1,0), c] += 1.0  # M3 to c01 (k=1, i=0)
        W[wrow(1,1), c] += 1.0  # M3 to c11 (k=1, i=1)
        
        # M4 = a11*(b00-b10)
        c = col_offset + 3
        U[urow(1,1), c] += 1.0
        V[vrow(0,0), c] += 1.0
        V[vrow(1,0), c] -= 1.0
        W[wrow(0,0), c] += 1.0  # M4 to c00
        W[wrow(0,1), c] += 1.0  # M4 to c10
        
        # M5 = (a00+a01)*b11
        c = col_offset + 4
        U[urow(0,0), c] += 1.0
        U[urow(0,1), c] += 1.0
        V[vrow(1,1), c] += 1.0
        W[wrow(0,0), c] -= 1.0  # -M5 to c00
        W[wrow(1,0), c] += 1.0  # +M5 to c01
        
        # M6 = (a10-a01)*(b01+b11)
        c = col_offset + 5
        U[urow(1,0), c] += 1.0
        U[urow(0,1), c] -= 1.0
        V[vrow(0,1), c] += 1.0
        V[vrow(1,1), c] += 1.0
        W[wrow(1,1), c] += 1.0  # M6 to c11
        
        # M7 = (a01-a10)*(b00+b10)
        c = col_offset + 6
        U[urow(0,1), c] += 1.0
        U[urow(1,0), c] -= 1.0
        V[vrow(0,0), c] += 1.0
        V[vrow(1,0), c] += 1.0
        W[wrow(0,0), c] += 1.0  # M7 to c00
        
        return col_offset + 7
    
    # Build the 22-rank decomposition
    rank_total = 22
    U = np.zeros((n * m, rank_total), dtype=float)  # (6, 22)
    V = np.zeros((m * p, rank_total), dtype=float)  # (18, 22)
    W = np.zeros((p * n, rank_total), dtype=float)  # (12, 22)
    
    # Part 1: 2x2 * 2x6 using 3 Strassen blocks
    # A rows 0-1, cols 0-1; B rows 0-1, cols 0-5; C rows 0-1, cols 0-5
    col = 0
    # Block 1: B cols 0-1
    col = build_strassen_2x2_2x2(U, V, W, col, 0, 0, 0)
    # Block 2: B cols 2-3
    col = build_strassen_2x2_2x2(U, V, W, col, 0, 0, 2)
    # Block 3: B cols 4-5
    col = build_strassen_2x2_2x2(U, V, W, col, 0, 0, 4)
    
    # Part 2: 2x1 * 1x6 (A col 2, B row 2)
    # C[i,k] = A[i,2]*B[2,k]
    # This is rank 1: U[i*m+2, c]=1 for i=0,1; V[2*p+k, c]=1 for k=0..5; W[k*n+i, c]=1
    c = col
    U[0 * m + 2, c] = 1.0  # A[0,2]
    U[1 * m + 2, c] = 1.0  # A[1,2]
    for k in range(p):
        V[2 * p + k, c] = 1.0  # B[2,k]
        W[k * n + 0, c] = 1.0  # C[k,0]
        W[k * n + 1, c] = 1.0  # C[k,1]
    col += 1
    
    assert col == rank_total, f"Column count mismatch: {col} vs {rank_total}"
    
    if verify(U, V, W, rank_total):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 36
    if time.time() - t_start > DEADLINE - 5:
        # Not enough time for fallback verification, just return schoolbook without full verify
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school), dtype=float)
        V_s = np.zeros((m * p, rank_school), dtype=float)
        W_s = np.zeros((p * n, rank_school), dtype=float)
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col_s] = 1.0
                    V_s[j * p + k, col_s] = 1.0
                    W_s[k * n + i, col_s] = 1.0
                    col_s += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    rank_school = n * m * p  # 36
    U_s = np.zeros((n * m, rank_school), dtype=float)
    V_s = np.zeros((m * p, rank_school), dtype=float)
    W_s = np.zeros((p * n, rank_school), dtype=float)
    col_s = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col_s] = 1.0
                V_s[j * p + k, col_s] = 1.0
                W_s[k * n + i, col_s] = 1.0
                col_s += 1
    
    if verify(U_s, V_s, W_s, rank_school):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    raise ValueError("No valid decomposition found")
# EVOLVE-BLOCK-END
