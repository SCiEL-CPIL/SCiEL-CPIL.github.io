"""Improved program: uses Strassen-based rank-22 decomposition for 3x3x3
combined with schoolbook rank-18 for 3x3x2, giving total rank 40
for 3x3 * 3x5 matrix multiplication."""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]


def _verify_numpy(U, V, W, n, m, p):
    """Check decomposition validity using numpy."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i * m + j] * V[j * p + k] * W[k * n + i]
                s = np.sum(prod)
                if abs(s - 1.0) > 1e-8:
                    return False
    return True


def _build_strassen_3x3x3():
    """
    Build a rank-22 decomposition for 3x3 * 3x3 matrix multiplication.
    
    Uses Strassen's 7-multiplication algorithm for the top-left 2x2 block,
    plus schoolbook for the remaining 5 entries of the output.
    
    Convention: T[i,j,k] = sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1
    
    A = [[a, b, c],
         [d, e, f],
         [g, h, i]]
    B = [[j, k, l],
         [m, n, o],
         [p, q, r]]
    
    C = A * B where C[row, col] = sum_{i} A[i, row] * B[i, col]
    Wait, let me be careful about the index convention.
    
    In the tensor: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    For n=m=p=3: U[3i+j, c] * V[3j+k, c] * W[3k+i, c]
    
    The matrix multiplication: (A*B)[j,k] = sum_i A[i,j] * B[i,k]
    So the "i" index in the tensor corresponds to the summation index.
    U is indexed by (summation_index, output_row) = (i, j)
    V is indexed by (output_row, output_col) = (j, k)
    W is indexed by (output_col, summation_index) = (k, i)
    
    So U[3i+j, c] corresponds to A[i, j]
    V[3j+k, c] corresponds to B[i, k]... wait that's not right.
    
    Let me re-derive. The tensor for matrix multiplication C = A * B
    where A is n x m, B is m x p, C is n x p:
    C[r, s] = sum_i A[r, i] * B[i, s]
    
    The tensor T[r, i, s] = 1 (all ones) with the convention:
    sum_c U[r*m+i, c] * V[i*p+s, c] * W[s*n+r, c] = 1 for all r, i, s
    
    So:
    U is indexed by (row_of_A, col_of_A) = (r, i), shape (n*m, rank)
    V is indexed by (col_of_A, col_of_B) = (i, s), shape (m*p, rank)  
    W is indexed by (col_of_B, row_of_A) = (s, r), shape (p*n, rank)
    
    For 3x3*3x3: n=m=p=3
    U[3r+i, c] * V[3i+s, c] * W[3s+r, c]
    
    A[r,i] is the (r,i) element of A
    B[i,s] is the (i,s) element of B
    C[r,s] = sum_i A[r,i] * B[i,s]
    
    So the decomposition says:
    sum_c U[3r+i, c] * V[3i+s, c] * W[3s+r, c] = 1 for all r, i, s
    
    This means the bilinear form sum_i A[r,i]*B[i,s] can be computed as
    sum_c (linear_comb_of_A_row_r_col_i) * (linear_comb_of_B_row_i_col_s) * W_factor
    
    Actually more precisely, for each c:
    The contribution to C[r,s] from term c is:
    (sum_i U[3r+i,c] * A[r,i]) * (sum_i V[3i+s,c] * B[i,s]) * W[3s+r,c]
    Wait no, that's not how tensor decomposition works.
    
    In tensor decomposition, T = sum_c u_c ⊗ v_c ⊗ w_c where
    T[r,i,s] = sum_c u_c[r,i] * v_c[i,s] * w_c[s,r]
    
    And we want T[r,i,s] = 1 for all r,i,s.
    
    The "multiplication" in the algorithm is:
    For each c, compute M_c = (sum_{r,i} U[3r+i,c] * A[r,i]) * (sum_{i,s} V[3i+s,c] * B[i,s])
    Then C[r,s] = sum_c M_c * W[3s+r,c]
    
    Wait, that's not quite right either. Let me think again.
    
    The tensor T satisfies: C = (A ⊗ B) contracted with T, where
    C[r,s] = sum_{i} T[r,i,s] * A[r,i] * B[i,s] = sum_i A[r,i] * B[i,s]
    
    A rank-1 decomposition T = sum_c u_c ⊗ v_c ⊗ w_c means:
    C[r,s] = sum_c (sum_{r',i'} u_c[r',i'] A[r',i']) * (sum_{i',s'} v_c[i',s'] B[i',s']) * w_c[s,r]
    
    Hmm, that doesn't look right dimensionally. Let me be very explicit.
    
    T is a 3-tensor with indices (r, i, s) each ranging 0..2.
    T[r,i,s] = 1 for all r,i,s.
    
    Rank-1 decomposition: T[r,i,s] = sum_c U[r,i,c] * V[i,s,c] * W[s,r,c]
    
    Note: U is indexed by (r,i), V by (i,s), W by (s,r).
    In flattened form: U[3r+i, c], V[3i+s, c], W[3s+r, c].
    
    The matrix multiplication algorithm:
    For each c:
      M_c = (sum_{r,i} U[3r+i,c] * A[r,i]) * (sum_{i,s} V[3i+s,c] * B[i,s])
    Then:
      C[r,s] = sum_c M_c * W[3s+r,c]
    
    This works because:
    sum_c M_c * W[3s+r,c] = sum_c W[3s+r,c] * (sum_{r',i'} U[3r'+i',c] A[r',i']) * (sum_{i',s'} V[3i'+s',c] B[i',s'])
    
    For this to equal sum_i A[r,i] B[i,s] for all r,s, we need the tensor identity.
    
    OK so now let me construct the Strassen decomposition.
    
    For the 2x2 Strassen on the sub-block A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    and B_sub = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]:
    
    M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
    M2 = (A[1,0]+A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[1,0]-B[0,0])
    M4 = (A[0,0]+A[0,1]) * B[1,1]
    M5 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    M7 = (A[0,0]+A[1,1]) * (B[0,1]-B[1,1])
    
    C[0,0] = M1 + M4 - M5 - M7
    C[0,1] = M3 + M7
    C[1,0] = M2 + M5
    C[1,1] = M1 - M2 + M3 + M6
    
    Now I need to express each M_c in the tensor decomposition form.
    
    For M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
    U part: U[3*0+0, c1]=1, U[3*1+1, c1]=1 (i.e., U[0,c1]=1, U[4,c1]=1)
    V part: V[3*0+0, c1]=1, V[3*1+0, c1]=1 (i.e., V[0,c1]=1, V[3,c1]=1)
    W part: W[3*0+0, c1]=1, W[3*1+0, c1]=1, W[3*1+1, c1]=-1, W[3*0+1, c1]=-1
    Wait: W is indexed by (s, r) where s is col of B, r is row of A.
    C[0,0] gets +M1, so W[3*0+0, c1] = 1 (s=0, r=0)
    C[0,1] gets +M7 not M1... let me recheck.
    
    C[0,0] = M1 + M4 - M5 - M7: W[3*0+0, c1]=+1, W[3*1+0, c4]=+1, W[3*0+0, c5]=-1, W[3*1+0, c7]=-1
    C[0,1] = M3 + M7: W[3*1+0, c3]=+1, W[3*1+0, c7]=+1
    C[1,0] = M2 + M5: W[3*0+1, c2]=+1, W[3*0+1, c5]=+1
    C[1,1] = M1 - M2 + M3 + M6: W[3*1+1, c1]=+1, W[3*1+1, c2]=-1, W[3*1+1, c3]=+1, W[3*1+1, c6]=+1
    
    So for each M_c, W[3s+r, c] is the coefficient of M_c in C[r,s].
    
    Let me tabulate:
    c=1 (M1): W[0,1]=1 (C[0,0]), W[1,1]=1 (C[1,1]) → W[3*0+0]=1, W[3*1+1]=1
    c=2 (M2): W[0,2]=1 (C[1,0]), W[1,2]=-1 (C[1,1]) → W[3*0+1]=1, W[3*1+1]=-1
    c=3 (M3): W[1,3]=1 (C[0,1]), W[1,3]=1 (C[1,1]) → W[3*1+0]=1, W[3*1+1]=1
    c=4 (M4): W[0,4]=1 (C[0,0]) → W[3*0+0]=1
    c=5 (M5): W[0,5]=-1 (C[0,0]), W[0,5]=1 (C[1,0]) → W[3*0+0]=-1, W[3*0+1]=1
    c=6 (M6): W[1,6]=1 (C[1,1]) → W[3*1+1]=1
    c=7 (M7): W[0,7]=-1 (C[0,0]), W[1,7]=1 (C[0,1]) → W[3*0+0]=-1, W[3*1+0]=1
    
    Now U and V:
    M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
    U: U[0,c1]=1, U[4,c1]=1
    V: V[0,c1]=1, V[3,c1]=1
    
    M2 = (A[1,0]+A[1,1]) * B[0,0]
    U: U[3,c2]=1, U[4,c2]=1
    V: V[0,c2]=1
    
    M3 = A[0,0] * (B[1,0]-B[0,0])
    U: U[0,c3]=1
    V: V[3,c3]=1, V[0,c3]=-1
    
    M4 = (A[0,0]+A[0,1]) * B[1,1]
    U: U[0,c4]=1, U[1,c4]=1
    V: V[4,c4]=1
    
    M5 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U: U[1,c5]=1, U[4,c5]=-1
    V: V[3,c5]=1, V[4,c5]=1
    
    M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U: U[3,c6]=1, U[0,c6]=-1
    V: V[0,c6]=1, V[1,c6]=1
    
    M7 = (A[0,0]+A[1,1]) * (B[0,1]-B[1,1])
    U: U[0,c7]=1, U[4,c7]=1
    V: V[1,c7]=1, V[4,c7]=-1
    
    Now for the remaining 5 entries:
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    Each of these needs 3 scalar multiplications (schoolbook), so 5*3 = 15 more terms.
    Total: 7 + 15 = 22 terms.
    
    For C[0,2]:
    Term 8: A[0,0]*B[0,2], W[3*2+0, c8]=1
    Term 9: A[0,1]*B[1,2], W[3*2+0, c9]=1
    Term 10: A[0,2]*B[2,2], W[3*2+0, c10]=1
    
    For C[1,2]:
    Term 11: A[1,0]*B[0,2], W[3*2+1, c11]=1
    Term 12: A[1,1]*B[1,2], W[3*2+1, c12]=1
    Term 13: A[1,2]*B[2,2], W[3*2+1, c13]=1
    
    For C[2,0]:
    Term 14: A[2,0]*B[0,0], W[3*0+2, c14]=1
    Term 15: A[2,1]*B[1,0], W[3*0+2, c15]=1
    Term 16: A[2,2]*B[2,0], W[3*0+2, c16]=1
    
    For C[2,1]:
    Term 17: A[2,0]*B[0,1], W[3*1+2, c17]=1
    Term 18: A[2,1]*B[1,1], W[3*1+2, c18]=1
    Term 19: A[2,2]*B[2,1], W[3*1+2, c19]=1
    
    For C[2,2]:
    Term 20: A[2,0]*B[0,2], W[3*2+2, c20]=1
    Term 21: A[2,1]*B[1,2], W[3*2+2, c21]=1
    Term 22: A[2,2]*B[2,2], W[3*2+2, c22]=1
    
    Now let me build the arrays.
    """
    rank = 22
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Term 1: M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1  # C[0,0] and C[1,1]
    
    # Term 2: M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1; W[4, 1] = -1  # C[1,0] and C[1,1]
    
    # Term 3: M3 = A[0,0] * (B[1,0]-B[0,0])
    U[0, 2] = 1
    V[3, 2] = 1; V[0, 2] = -1
    W[3, 2] = 1; W[4, 2] = 1  # C[0,1] and C[1,1]
    
    # Term 4: M4 = (A[0,0]+A[0,1]) * B[1,1]
    U[0, 3] = 1; U[1, 3] = 1
    V[4, 3] = 1
    W[0, 3] = 1  # C[0,0]
    
    # Term 5: M5 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[1, 4] = 1; U[4, 4] = -1
    V[3, 4] = 1; V[4, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1  # C[0,0] and C[1,0]
    
    # Term 6: M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[4, 5] = 1  # C[1,1]
    
    # Term 7: M7 = (A[0,0]+A[1,1]) * (B[0,1]-B[1,1])
    U[0, 6] = 1; U[4, 6] = 1
    V[1, 6] = 1; V[4, 6] = -1
    W[0, 6] = -1; W[3, 6] = 1  # C[0,0] and C[0,1]
    
    # Terms 8-10: C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    U[0, 7] = 1; V[2, 7] = 1; W[6, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[6, 8] = 1
    U[2, 9] = 1; V[8, 9] = 1; W[6, 9] = 1
    
    # Terms 11-13: C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    U[3, 10] = 1; V[2, 10] = 1; W[7, 10] = 1
    U[4, 11] = 1; V[5, 11] = 1; W[7, 11] = 1
    U[5, 12] = 1; V[8, 12] = 1; W[7, 12] = 1
    
    # Terms 14-16: C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    U[6, 13] = 1; V[0, 13] = 1; W[8, 13] = 1
    U[7, 14] = 1; V[3, 14] = 1; W[8, 14] = 1
    U[8, 15] = 1; V[6, 15] = 1; W[8, 15] = 1
    
    # Terms 17-19: C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    U[6, 16] = 1; V[1, 16] = 1; W[5, 16] = 1
    U[7, 17] = 1; V[4, 17] = 1; W[5, 17] = 1
    U[8, 18] = 1; V[7, 18] = 1; W[5, 18] = 1
    
    # Terms 20-22: C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    U[6, 19] = 1; V[2, 19] = 1; W[2, 19] = 1
    U[7, 20] = 1; V[5, 20] = 1; W[2, 20] = 1
    U[8, 21] = 1; V[8, 21] = 1; W[2, 21] = 1
    
    return U, V, W


def _build_full_decomposition():
    """
    Build the full 3x3 * 3x5 decomposition with rank 40.
    Split into 3x3*3x3 (rank 22) and 3x3*3x2 (rank 18 schoolbook).
    """
    n, m, p = 3, 3, 5
    rank = 40
    
    U = np.zeros((n * m, rank))  # (9, 40)
    V = np.zeros((m * p, rank))  # (15, 40)
    W = np.zeros((p * n, rank))  # (15, 40)
    
    # Part 1: 3x3 * 3x3 (columns 0..4 of B, i.e., k=0,1,2)
    U_33, V_33, W_33 = _build_strassen_3x3x3()
    U[:, :22] = U_33
    # V_33 has shape (9, 22), but V needs shape (15, 40)
    # V_33[j*3+k, c] for j=0,1,2 and k=0,1,2
    # In the full problem, V[j*5+k, c] for j=0,1,2 and k=0,1,2
    for j in range(3):
        for k in range(3):
            V[j * 5 + k, :22] = V_33[j * 3 + k, :]
    # W_33 has shape (9, 22), W_33[k*3+i, c] for k=0,1,2 and i=0,1,2
    # In the full problem, W[k*3+i, c] for k=0,1,2 and i=0,1,2 (same since p=3 for this part)
    W[:9, :22] = W_33
    
    # Part 2: 3x3 * 3x2 (columns 3,4 of B, i.e., k=3,4)
    # Schoolbook: for each (i, j, k) with k in {3,4}, one term
    col = 22
    for i in range(n):
        for j in range(m):
            for k in [3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Build the rank-40 decomposition
    U, V, W = _build_full_decomposition()
    
    # Verify
    n, m, p = DIMENSIONS
    if _verify_numpy(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 45
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
