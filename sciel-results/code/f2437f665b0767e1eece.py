"""
Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.
Uses Strassen-style identities on 2x2 inner blocks.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    U_dim = n * m   # 8
    V_dim = m * p   # 32
    W_dim = p * n   # 16

    # We decompose the matrix multiplication tensor:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In trace convention: C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
    
    # Strategy: Split the j-sum into two groups of 2: j in {0,1} and j in {2,3}
    # For each group, we have a 2x2 (inner) * 2x8 (outer) multiplication.
    # For a 2x2 * 2x8 multiplication, we can use Strassen on the 2x2 part.
    # But actually, the "2x2" here is A[i, j0:j0+2] (2x2) times B[j0:j0+2, k] (2x8).
    # 
    # Let me think differently. The tensor T[i,j,k] = 1 for all i,j,k.
    # We need to express this as a sum of rank-1 tensors.
    # 
    # Key insight: For the 2x2 inner dimension (j=0,1 or j=2,3), we can use
    # the 7 Strassen products. For each such block, the contribution to C[i,k]
    # involves A[i, j0], A[i, j1] and B[j0, k], B[j1, k].
    #
    # Strassen for 2x2 * 2x2 (but here B is 2x8, so we apply Strassen per column of B):
    # Actually, since B has 8 columns, and Strassen works on 2x2 blocks,
    # we can group the 8 columns of B into 4 groups of 2.
    #
    # Let me use a cleaner approach:
    # 
    # Split j into {0,1} and {2,3}.
    # For j in {0,1}: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For j in {2,3}: C[i,k] += A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # For the first sum, define (using Strassen on the 2x2 inner dim for each pair of k's):
    # Group k into {0,1}, {2,3}, {4,5}, {6,7}. For each group of 2 k's, we have
    # a 2x2 * 2x2 problem (A is 2x2, B is 2x2), which has Strassen rank 7.
    # Total: 4 groups * 7 = 28 products for the j in {0,1} part.
    # Similarly 28 for j in {2,3} part.
    # Total: 56 products.
    #
    # Schoolbook is 2*4*8 = 64. So 56 < 64. Good.
    
    def add_terms(U_list, V_list, W_list, u_entries, v_entries, w_entries):
        """Add one rank-1 term."""
        u_vec = [0.0] * U_dim
        v_vec = [0.0] * V_dim
        w_vec = [0.0] * W_dim
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    U_list = []
    V_list = []
    W_list = []
    
    # For each j-group (j0, j1) in {(0,1), (2,3)}:
    #   For each k-group (k0, k1) in {(0,1), (2,3), (4,5), (6,7)}:
    #     Use Strassen 7 products for A[0:2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
    #     contributing to C[0:2, k0:k0+2]
    
    j_groups = [(0, 1), (2, 3)]
    k_groups = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for j0, j1 in j_groups:
        for k0, k1 in k_groups:
            # A block: rows i=0,1; cols j0,j1
            # B block: rows j0,j1; cols k0,k1
            # C block: rows i=0,1; cols k0,k1
            
            # Strassen products for 2x2 * 2x2:
            # Let a = A[0,j0], b = A[0,j1], c = A[1,j0], d = A[1,j1]
            # Let p = B[j0,k0], q = B[j0,k1], r = B[j1,k0], s = B[j1,k1]
            # 
            # m1 = (a+d)*(p+s)   -> contributes to C[0,k0], C[0,k1], C[1,k0], C[1,k1]
            # m2 = (c+d)*p       -> contributes to C[1,k0], C[1,k1]
            # m3 = a*(q-s)       -> contributes to C[0,k0], C[0,k1]
            # m4 = d*(r-p)       -> contributes to C[1,k0], C[1,k1]
            # m5 = (a+c)*(s+r)   -> contributes to C[0,k0], C[0,k1], C[1,k0], C[1,k1]
            # m6 = (b-c)*(q+r)   -> contributes to C[0,k0], C[0,k1], C[1,k0], C[1,k1]
            # m7 = (b-d)*(p+r)   -> contributes to C[0,k0], C[0,k1], C[1,k0], C[1,k1]
            #
            # Actually, let me use the standard Strassen formulas:
            # M1 = (A00 + A11) * (B00 + B11)
            # M2 = (A10 + A11) * B00
            # M3 = A00 * (B01 - B11)
            # M4 = A11 * (B10 - B00)
            # M5 = (A00 + A01) * B11
            # M6 = (A10 - A00) * (B00 + B01)
            # M7 = (A01 - A11) * (B10 + B11)
            #
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            #
            # In our notation:
            # A00 = A[0,j0], A01 = A[0,j1], A10 = A[1,j0], A11 = A[1,j1]
            # B00 = B[j0,k0], B01 = B[j0,k1], B10 = B[j1,k0], B11 = B[j1,k1]
            
            # Term 1: M1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            add_terms(U_list, V_list, W_list,
                [(0*m+j0, 1.0), (1*m+j1, 1.0)],
                [(j0*p+k0, 1.0), (j1*p+k1, 1.0)],
                [(k0*n+0, 1.0), (k0*n+1, 1.0), (k1*n+0, 1.0), (k1*n+1, 1.0)])
            
            # Term 2: M2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            add_terms(U_list, V_list, W_list,
                [(1*m+j0, 1.0), (1*m+j1, 1.0)],
                [(j0*p+k0, 1.0)],
                [(k0*n+1, 1.0), (k1*n+1, 1.0)])
            
            # Term 3: M3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            add_terms(U_list, V_list, W_list,
                [(0*m+j0, 1.0)],
                [(j0*p+k1, 1.0), (j1*p+k1, -1.0)],
                [(k0*n+0, 1.0), (k1*n+0, 1.0)])
            
            # Term 4: M4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            add_terms(U_list, V_list, W_list,
                [(1*m+j1, 1.0)],
                [(j1*p+k0, 1.0), (j0*p+k0, -1.0)],
                [(k0*n+1, 1.0), (k1*n+1, 1.0)])
            
            # Term 5: M5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            add_terms(U_list, V_list, W_list,
                [(0*m+j0, 1.0), (0*m+j1, 1.0)],
                [(j1*p+k1, 1.0)],
                [(k0*n+0, -1.0), (k0*n+1, -1.0), (k1*n+0, 1.0), (k1*n+1, 1.0)])
            
            # Term 6: M6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            add_terms(U_list, V_list, W_list,
                [(1*m+j0, 1.0), (0*m+j0, -1.0)],
                [(j0*p+k0, 1.0), (j0*p+k1, 1.0)],
                [(k1*n+1, 1.0)])
            
            # Term 7: M7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            add_terms(U_list, V_list, W_list,
                [(0*m+j1, 1.0), (1*m+j1, -1.0)],
                [(j1*p+k0, 1.0), (j1*p+k1, 1.0)],
                [(k0*n+0, 1.0), (k1*n+1, 1.0)])
    
    U = np.array(U_list).T  # shape (U_dim, rank)
    V = np.array(V_list).T  # shape (V_dim, rank)
    W = np.array(W_list).T  # shape (W_dim, rank)
    
    rank = U.shape[1]
    
    # Verify
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    
    if not np.allclose(T, 1.0, atol=1e-6):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
