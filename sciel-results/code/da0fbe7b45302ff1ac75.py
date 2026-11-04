"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = 1 if output (i,k) depends on input (j)
    # T is indexed by (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for efficient computation
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                # The tensor entry is 1 (each output element is sum over j of A[i,j]*B[j,k])
                # Actually T[i,j,k] = 1 for all i,j,k since C[i,k] = sum_j A[i,j]*B[j,k]
                if abs(val - 1.0) > tol:
                    errors.append((i, j, k, val))
    return len(errors) == 0, errors

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    # Strassen's algorithm for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    # 
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c-d)(f+g)
    # m7 = (b-d)(e+g)
    #
    # C[0,0] = m1 + m4 - m5 - m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m7
    #
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2
    # U row index: i*2+j, V row index: j*2+k, W row index: k*2+i
    
    U = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    V = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    W = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    
    # m1 = (a+d)(e+h): A entries (0,0) and (1,1), B entries (0,0) and (1,1)
    # U: rows 0 (a) and 2 (d) get +1
    # V: rows 0 (e) and 3 (h) get +1
    # W: contributes to C[0,0], C[1,1] -> W[0*2+0]=W[0] (C[0,0]), W[1*2+1]=W[3] (C[1,1])
    U[0][0] += 1; U[2][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1  # C[0,0]
    W[3][0] += 1  # C[1,1]
    
    # m2 = (c+d)e: A entries (1,0) and (1,1), B entry (0,0)
    # U: rows 1 (c) and 2 (d) get +1
    # V: row 0 (e) gets +1
    # W: contributes to C[1,0] -> W[0*2+1]=W[1]
    U[1][1] += 1; U[2][1] += 1
    V[0][1] += 1
    W[1][1] += 1  # C[1,0]
    
    # m3 = a(f-h): A entry (0,0), B entries (0,1) and (1,1)
    # U: row 0 (a) gets +1
    # V: row 1 (f) gets +1, row 3 (h) gets -1
    # W: contributes to C[0,1] and C[1,1] -> W[1*2+0]=W[2], W[1*2+1]=W[3]
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[2][2] += 1  # C[0,1]
    W[3][2] += 1  # C[1,1]
    
    # m4 = d(g-c): A entries (1,1) and (1,0), B entries (1,0) and (0,0)
    # U: row 2 (d) gets +1, row 1 (c) gets -1
    # V: row 2 (g) gets +1, row 0 (e) gets -1
    # W: contributes to C[0,0] and C[1,0] -> W[0], W[1]
    U[2][3] += 1; U[1][3] -= 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1  # C[0,0]
    W[1][3] += 1  # C[1,0]
    
    # m5 = (a+b)h: A entries (0,0) and (0,1), B entry (1,1)
    # U: row 0 (a) gets +1, row 1 (b) gets +1
    # V: row 3 (h) gets +1
    # W: contributes to C[0,0] and C[0,1] -> W[0], W[2]
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1  # C[0,0] (negative in formula)
    W[2][4] += 1  # C[0,1]
    
    # m6 = (c-d)(f+g): A entries (1,0) and (1,1), B entries (0,1) and (1,0)
    # U: row 1 (c) gets +1, row 2 (d) gets -1
    # V: row 1 (f) gets +1, row 2 (g) gets +1
    # W: not used in standard Strassen (it's there for completeness but not needed for 2x2x2)
    # Actually m6 is not used in the 4 output terms. Let me re-check.
    # C[0,0] = m1+m4-m5-m7, C[0,1]=m3+m5, C[1,0]=m2+m4, C[1,1]=m1-m2+m3+m7
    # m6 is indeed not used. So we can skip it or set it to 0.
    U[1][5] += 1; U[2][5] -= 1
    V[1][5] += 1; V[2][5] += 1
    # No W contribution
    
    # m7 = (b-d)(e+g): A entries (0,1) and (1,1), B entries (0,0) and (1,0)
    # U: row 1 (b) gets +1, row 2 (d) gets -1
    # V: row 0 (e) gets +1, row 2 (g) gets +1
    # W: contributes to C[0,0] and C[1,1] -> W[0], W[3]
    U[1][6] += 1; U[2][6] -= 1
    V[0][6] += 1; V[2][6] += 1
    W[0][6] -= 1  # C[0,0] (negative)
    W[3][6] += 1  # C[1,1]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Strategy: Partition m=3 into m1=2 and m2=1
    # Block 1: 2x2x6 sub-problem
    # Block 2: 2x1x6 sub-problem
    
    # For 2x2x6: We can use Strassen on the 2x2 part and handle the 6 columns
    # 6 = 2*3, so we can view B as 2x(2*3) and use Kronecker structure
    # 
    # Actually, let's think differently. For 2x2x6:
    # We can partition the 6 columns into 3 groups of 2, and apply Strassen to each pair
    # That gives 3 * 7 = 21 terms for the 2x2x6 block.
    #
    # For 2x1x6: This is just matrix-vector multiplication (2x1 times 1x6), rank = 2*1*6 = 12
    # But we can do better: rank of 2x1x6 tensor. The tensor T[i,j,k] where j is always 0.
    # T[i,0,k] = 1 for all i,k. This is an outer product of a 2-vector and a 6-vector,
    # so rank = 1? No, wait. The tensor for matrix mult 2x1x6:
    # C[i,k] = A[i,0]*B[0,k]. So T[i,0,k] = 1.
    # This tensor has rank 1? No, it's a 2x1x6 tensor. 
    # Actually the multiplication tensor for (n x 1) * (1 x p) has rank n*p because
    # C[i,k] = A[i,0]*B[0,k] and these are independent.
    # Wait no: T[i,0,k] = 1 for all i,k. The tensor is just the all-ones tensor of shape 2x1x6.
    # The rank of the all-ones tensor of shape a x b x c is... 
    # Actually for matrix multiplication, the tensor is T[i,j,k] where the multiplication is
    # C[i,k] = sum_j A[i,j] B[j,k]. For n=2,m=1,p=6: C[i,k] = A[i,0]*B[0,k].
    # The tensor entries are T[i,0,k] = 1 for all i in {0,1}, k in {0,...,5}.
    # This is a rank-1 tensor! Because T[i,0,k] = 1*1*1 = u_i * v_0 * w_k where
    # u = [1,1], v = [1], w = [1,1,1,1,1,1].
    # Wait, that's not right either. The tensor for bilinear map is T in U*V*W* such that
    # T(a,b) = sum T[i,j,k] a[i,j] b[j,k].
    # For n=2,m=1,p=6: T[i,0,k] = 1 means the bilinear form is sum_{i,k} A[i,0]*B[0,k]*C[i,k].
    # The tensor is sum_{i=0}^{1} sum_{k=0}^{5} e_i ⊗ e_0 ⊗ e_k.
    # This can be written as (sum_i e_i) ⊗ e_0 ⊗ (sum_k e_k), which is rank 1!
    # So rank(2x1x6) = 1? Let me verify: U has shape 2*1=2, V has shape 1*6=6, W has shape 6*2=12.
    # U[i, c] for i in {0,1}, V[0*6+k, c] for k in {0,...,5}, W[k*2+i, c].
    # For rank 1: U[i,0] = 1 for i=0,1; V[k,0] = 1 for k=0..5; W[k*2+i,0] = 1 for all k,i.
    # Check: sum_c U[i,0] V[k,0] W[k*2+i,0] = 1*1*1 = 1 = T[i,0,k]. Yes! Rank 1.
    
    # So total: 21 (Strassen for 2x2x6) + 1 (for 2x1x6) = 22?
    # But wait, we need to be careful about how the blocks interact.
    
    # Let me reconsider. The full problem is 2x3x6.
    # A is 2x3, B is 3x6, C is 2x6.
    # C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k]
    # = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    
    # Partition: j in {0,1} handled by 2x2x6 block, j=2 handled by 2x1x6 block.
    # For the 2x2x6 block: C1[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # For the 2x1x6 block: C2[i,k] = A[i,2]B[2,k]
    # C = C1 + C2
    
    # For 2x2x6: partition the 6 columns into 3 pairs: (0,1), (2,3), (4,5)
    # For each pair of columns, we have a 2x2x2 sub-problem (2 rows, 2 cols of B, 2 cols of C)
    # Apply Strassen to each: 3 * 7 = 21 terms.
    
    # For 2x1x6: rank 1 as shown above.
    
    # Total rank: 21 + 1 = 22.
    
    # But wait - can we do better? The known lower bound for 2x3x6...
    # Let me just go with 22 and verify.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Block 1: 2x2x6 via 3 Strassen instances on column pairs
    for group in range(3):
        k_start = group * 2
        sU, sV, sW = _strassen_2x2x2()
        # sU: 4 rows (i*2+j for i in {0,1}, j in {0,1}), 7 cols
        # sV: 4 rows (j*2+k for j in {0,1}, k in {0,1}), 7 cols
        # sW: 4 rows (k*2+i for k in {0,1}, i in {0,1}), 7 cols
        #
        # In our 2x2x6 block, the A indices are (i, j) where i in {0,1}, j in {0,1} (global j=0,1)
        # The B indices are (j, k) where j in {0,1}, k in {k_start, k_start+1}
        # The C indices are (i, k) where i in {0,1}, k in {k_start, k_start+1}
        #
        # sU row index: i*2 + j_local where j_local in {0,1} corresponds to global j = j_local
        # In the full U matrix (2*3=6 rows), row index is i*3 + j_global
        # So sU row (i*2+j_local) maps to full row i*3 + j_local
        #
        # sV row index: j_local*2 + k_local where k_local in {0,1} corresponds to global k = k_start + k_local
        # In the full V matrix (3*6=18 rows), row index is j_global*6 + k_global
        # So sV row (j_local*2+k_local) maps to full row j_local*6 + (k_start + k_local)
        #
        # sW row index: k_local*2 + i where k_local in {0,1} corresponds to global k = k_start + k_local
        # In the full W matrix (6*2=12 rows), row index is k_global*2 + i
        # So sW row (k_local*2+i) maps to full row (k_start+k_local)*2 + i
        
        rank_s = 7
        for c in range(rank_s):
            u_vals = [0]*6  # 2*3 rows
            v_vals = [0]*18  # 3*6 rows
            w_vals = [0]*12  # 6*2 rows
            for i in range(2):
                for j_local in range(2):
                    u_vals[i*3 + j_local] = sU[i*2 + j_local][c]
            for j_local in range(2):
                for k_local in range(2):
                    v_vals[j_local*6 + (k_start + k_local)] = sV[j_local*2 + k_local][c]
            for k_local in range(2):
                for i in range(2):
                    w_vals[(k_start + k_local)*2 + i] = sW[k_local*2 + i][c]
            U_rows.append(u_vals)
            V_rows.append(v_vals)
            W_rows.append(w_vals)
    
    # Block 2: 2x1x6, rank 1
    u_vals = [0]*6
    v_vals = [0]*18
    w_vals = [0]*12
    for i in range(2):
        u_vals[i*3 + 2] = 1  # j_global = 2
    for k in range(6):
        v_vals[2*6 + k] = 1  # j_global = 2
    for k in range(6):
        for i in range(2):
            w_vals[k*2 + i] = 1
    U_rows.append(u_vals)
    V_rows.append(v_vals)
    W_rows.append(w_vals)
    
    rank = len(U_rows)
    
    # Verify
    valid, errors = _verify(U_rows, V_rows, W_rows, n, m, p)
    if valid:
        return {"U": U_rows, "V": V_rows, "W": W_rows}
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
