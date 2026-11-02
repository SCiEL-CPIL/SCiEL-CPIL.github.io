import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    # Dimensions of tensors:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # The tensor is of shape (n*m, m*p, p*n)
    # Specifically: T[i*m+j, j'*p+k, k*n+i] = delta_{j, j'}
    
    # We want to minimize the rank.
    # The tensor can be decomposed into m=4 blocks for j=0..3.
    # For each j, the block T_j[i, j, k, j, k, i] = delta_{j,j} = 1? No.
    # T[i, j, j', k] = delta_{j, j'} for fixed i, k?
    # The indices are:
    # Row: A's index (i, j) with i in 0..n-1, j in 0..m-1. Total n*m = 8.
    # Col: B's index (j', k) with j' in 0..m-1, k in 0..p-1. Total m*p = 28.
    # Slice: C's index (k, i) with k in 0..p-1, i in 0..n-1. Total p*n = 14.
    # The entry T[ (i,j), (j',k), (k,i) ] = 1 if j == j' else 0.
    
    # For each (k, i) slice (14 of them), we have a matrix of size (8, 28).
    # Let's call this matrix S_{ki}.
    # S_{ki} [i*m+j, j'*p+k'] = 1 if k==k' and j==j' else 0.
    # Notice that S_{ki} is non-zero only at positions (i*m+j, j*p+k).
    # This is a permutation-like matrix.
    # Specifically, S_{ki} has a 1 at row r = i*m+j, col c = j*p+k for j=0..3.
    # It has rank 4 for each (k, i).
    # However, the tensor is rank min r such that T = sum_{alpha=0}^{r-1} u_alpha \otimes v_alpha \otimes w_alpha.
    
    # Let's try to find a low rank decomposition.
    # Since n=2, m=4, p=7.
    # Schoolbook rank is 56.
    # Strassen on 2x2 blocks:
    # We can group the 4 columns of A (m=4) into two pairs? Or group i?
    # Let's group i (size 2) and use Strassen-like 2x2x?
    # Actually, we can view the multiplication as:
    # C[k, i] = sum_{j,j'} A[i, j] B[j', k] delta_{j, j'} = sum_j A[i, j] B[j, k].
    # This is just standard matrix multiplication (2x4) * (4x7).
    # So it's just 2x4 times 4x7.
    # Can we beat schoolbook 56?
    # Yes, by blocking.
    # Split the 4 intermediate dimension j into two blocks of 2.
    # j in {0,1} and j in {2,3}.
    # C[k, i] = sum_{j=0}^1 A[i, j] B[j, k] + sum_{j=2}^3 A[i, j-2] B[j-2, k].
    # This is two independent matrix multiplications of size (2x2) * (2x7).
    # Rank(2x2 * 2x7) = ?
    # Let's compute rank of (2x2) * (2x7).
    # n=2, m=2, p=7.
    # Schoolbook is 28.
    # We can use Laderman's algorithm for 2x2x2 (rank 7).
    # Split p=7 into 2+2+2+1.
    # C[:, 0:2] uses Strassen 2x2x2 (rank 7).
    # C[:, 2:4] uses Strassen 2x2x2 (rank 7).
    # C[:, 4:6] uses Strassen 2x2x2 (rank 7).
    # C[:, 6:7] uses Schoolbook 2x2x1 (rank 4).
    # Total for one block of size (2x2) * (2x7) is 7+7+7+4 = 25.
    # Since we have 2 such blocks (for j=0,1 and j=2,3), total rank = 25 + 25 = 50.
    # This matches the previous score (50).
    
    # Can we do better?
    # Maybe combine the two (2x2)*(2x7) blocks?
    # We have A of shape (2, 4) and B of shape (4, 7).
    # C = A B.
    # We used rank 50.
    # Is there a better way?
    # What if we don't split j?
    # We can split p=7 into 2+2+3?
    # Or use a different blocking for the full (2x4) * (4x7).
    
    # Let's try to verify rank 50 solution first, as it is known to work.
    # Construction for one (2x2) * (2x7) block:
    # Inputs: A_block (2, 2), B_block (2, 7).
    # Output: C_block (7, 2). Note: C[k, i] so C is (p, n).
    
    # Let's implement the rank 50 construction carefully.
    
    # Helper for 2x2x2 Strassen
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A B
    # Rank 7 terms:
    # M1 = (a+d)(e+h) -> C[0,0] += M1, C[1,1] += M1
    # M2 = (c+d)e     -> C[1,0] += M2, C[1,1] -= M2
    # M3 = a(f-h)     -> C[0,1] += M3, C[1,1] += M3
    # M4 = d(g-e)     -> C[0,0] += M4, C[1,0] += M4
    # M5 = (a+b)h     -> C[0,0] -= M5, C[0,1] += M5
    # M6 = (c-a)(e+f) -> C[1,1] += M6
    # M7 = (b-d)(g+h) -> C[0,0] += M7
    
    # Let's assemble the full tensors U, V, W.
    # U has shape (2*4, rank). Indices (i, j) for A.
    # V has shape (4*7, rank). Indices (j, k) for B.
    # W has shape (7*2, rank). Indices (k, i) for C.
    
    rank_target = 50
    U = np.zeros((8, rank_target))
    V = np.zeros((28, rank_target))
    W = np.zeros((14, rank_target))
    
    col = 0
    
    def add_strassen_222(i_base, j_base, k_base):
        # Adds a Strassen block for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]
        nonlocal col
        # Indices in U: i is i_base, i_base+1; j is j_base, j_base+1
        u00 = i_base*4 + j_base
        u01 = i_base*4 + j_base + 1
        u10 = (i_base+1)*4 + j_base
        u11 = (i_base+1)*4 + j_base + 1
        
        # Indices in V: j is j_base, j_base+1; k is k_base, k_base+1
        v00 = j_base*7 + k_base
        v01 = j_base*7 + k_base + 1
        v10 = (j_base+1)*7 + k_base
        v11 = (j_base+1)*7 + k_base + 1
        
        # Indices in W: k is k_base, k_base+1; i is i_base, i_base+1
        # W[k*n + i] -> W[k*2 + i]
        w00 = k_base*2 + i_base
        w01 = k_base*2 + (i_base+1)
        w10 = (k_base+1)*2 + i_base
        w11 = (k_base+1)*2 + (i_base+1)
        
        # M1
        U[u00, col] += 1; U[u11, col] += 1
        V[v00, col] += 1; V[v11, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        # M2
        U[u10, col] += 1; U[u11, col] += 1
        V[v00, col] += 1
        W[w10, col] += 1; W[w11, col] -= 1
        col += 1
        # M3
        U[u00, col] += 1
        V[v01, col] += 1; V[v11, col] -= 1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        # M4
        U[u11, col] += 1
        V[v10, col] += 1; V[v00, col] -= 1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        # M5
        U[u00, col] += 1; U[u01, col] += 1
        V[v11, col] += 1
        W[w00, col] -= 1; W[w01, col] += 1
        col += 1
        # M6
        U[u10, col] += 1; U[u00, col] -= 1
        V[v00, col] += 1; V[v01, col] += 1
        W[w11, col] += 1
        col += 1
        # M7
        U[u01, col] += 1; U[u11, col] -= 1
        V[v10, col] += 1; V[v11, col] += 1
        W[w00, col] += 1
        col += 1
        
    def add_schoolbook_221(i_base, j_base, k_base):
        # Adds schoolbook for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+1]
        nonlocal col
        u00 = i_base*4 + j_base
        u01 = i_base*4 + j_base + 1
        u10 = (i_base+1)*4 + j_base
        u11 = (i_base+1)*4 + j_base + 1
        
        v00 = j_base*7 + k_base
        v10 = (j_base+1)*7 + k_base
        
        w00 = k_base*2 + i_base
        w01 = k_base*2 + (i_base+1)
        
        # Term 1: A00 B00 -> C00
        U[u00, col] += 1; V[v00, col] += 1; W[w00, col] += 1
        col += 1
        # Term 2: A01 B10 -> C01 (Wait, C index is k, i. Here k=k_base, i=i_base+1)
        # A[0,1] B[1,0] contributes to C[k_base, i_base+1]
        U[u01, col] += 1; V[v10, col] += 1; W[w01, col] += 1
        col += 1
        # Term 3: A10 B00 -> C10 (k=k_base, i=i_base+1? No, A[1,0] B[0,0] -> C[k_base, i_base+1])
        # Wait, C[i, k] = sum A[i, j] B[j, k].
        # Here i is the second index of C in W (k*n+i).
        # W[k, i].
        # A[i, j] B[j, k].
        # If i=1 (i_base+1), j=0 (j_base), k=k_base:
        # A[1,0] is u10. B[0, k_base] is v00. W[k_base, 1] is w01.
        U[u10, col] += 1; V[v00, col] += 1; W[w01, col] += 1
        col += 1
        # Term 4: A11 B10 -> C11 (k=k_base, i=i_base+1)
        # A[1,1] is u11. B[1, k_base] is v10. W[k_base, 1] is w01.
        U[u11, col] += 1; V[v10, col] += 1; W[w01, col] += 1
        col += 1
        
    # Apply to first block of j (j=0,1) and i=0,1
    # p=7 split into 2,2,2,1
    for k_start in [0, 2, 4]:
        add_strassen_222(0, 0, k_start)
    add_schoolbook_221(0, 0, 6)
    
    # Apply to second block of j (j=2,3) and i=0,1
    for k_start in [0, 2, 4]:
        add_strassen_222(0, 2, k_start)
    add_schoolbook_221(0, 2, 6)
    
    assert col == rank_target, f"Rank mismatch: {col} vs {rank_target}"
    
    # Verify correctness
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True
        
    if not verify(U, V, W):
        raise ValueError("Construction failed verification")

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
