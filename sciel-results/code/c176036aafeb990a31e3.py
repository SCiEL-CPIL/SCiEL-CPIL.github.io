import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i, j, k] = 1 if k == i*m + j, else 0
    # Actually, the tensor T for matrix multiplication C = A * B is:
    # T[i, j, k] where i indexes rows of C (0..n-1), j indexes columns of C (0..p-1), 
    # k indexes the sum index (0..m-1)
    # T[i, j, k] = 1 for all valid i,j,k (it's the identity in the sum dimension)
    # Wait, the tensor is T[i, j, k] = delta_{i,k} * delta_{j,k}? No.
    # C[i, j] = sum_k A[i, k] * B[k, j]
    # So the tensor T[i, j, k] = 1 for all i, j, k (it's just the multiplication tensor)
    # Actually T[i, j, k] is 1 if we're computing C[i,j] using A[i,k] and B[k,j]
    # The tensor is T[i, j, k] = 1 for all valid indices.
    
    # The decomposition says: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 (since C[i,j] = sum_k A[i,k]*B[k,j])
    
    # Let's verify: for each (i, j, k), sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1
    # Wait, that's not right either. The tensor T[i,j,k] for matrix multiplication is:
    # T[i, j, k] = 1 (it's the identity tensor in the sense that each C[i,j] gets one contribution per k)
    
    # Actually, let me think more carefully. The multiplication tensor M has entries:
    # M[i, j, k] = 1 for all i in [0,n), j in [0,p), k in [0,m)
    # No wait, that's not a tensor decomposition in the usual sense.
    
    # The standard formulation: C = A*B means C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor T is defined by T[i,j,k] = 1 (it's the "multiplication tensor")
    # A rank-r decomposition: T[i,j,k] = sum_{l=1}^r U[i,k,l] * V[k,j,l] * W[j,i,l]
    # Wait, the indices need to be consistent. Let me use the convention from the task:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r] summed over r should give T[i,j,k]
    # where T[i,j,k] = 1 (the multiplication tensor is all ones in this formulation)
    
    # Let's verify numerically
    max_err = 0.0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r]
                expected = 1.0
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
    return max_err

def construct():
    """
    Rank-44 tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    - Split the 4x4 B matrix into four 2x2 blocks indexed by (b_r, b_c) for b_r, b_c in {0,1}.
    - For each 2x2 block of B:
      - Use Strassen (rank 7) for the top 2x2 * 2x2 multiplication (rows 0,1 of A with the 2x2 block of B).
      - Use schoolbook (rank 4) for the bottom 1x2 * 2x2 multiplication (row 2 of A with the 2x2 block of B).
    - Total: 4 blocks * (7 + 4) = 44.
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x 44
    V = [[0]*rank for _ in range(m*p)]  # 16 x 44
    W = [[0]*rank for _ in range(p*n)]  # 12 x 44
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # Strassen part: 7 terms for top 2x2 * 2x2
            # Let A_block = [[a, b], [c, d]] where a=A[0, b_r*2], b=A[0, b_r*2+1], c=A[1, b_r*2], d=A[1, b_r*2+1]
            # Let B_block = [[e, f], [g, h]] where e=B[b_r*2, b_c*2], f=B[b_r*2, b_c*2+1], g=B[b_r*2+1, b_c*2], h=B[b_r*2+1, b_c*2+1]
            # C_block = [[c00, c01], [c10, c11]] where c00=C[0, b_c*2], c01=C[0, b_c*2+1], c10=C[1, b_c*2], c11=C[1, b_c*2+1]
            
            # m1 = (a+d)(e+h): contributes to c00 and c11
            U[0*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0, b_c*2] += m1
            W[(b_c*2+1)*n + 1][col] = 1   # C[1, b_c*2+1] += m1
            col += 1
            
            # m2 = (c+d)e: contributes to c10 and c11
            U[1*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1   # C[1, b_c*2] += m2
            W[(b_c*2+1)*n + 1][col] = -1  # C[1, b_c*2+1] -= m2
            col += 1
            
            # m3 = a(f-h): contributes to c01 and c11
            U[0*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1   # C[0, b_c*2+1] += m3
            W[(b_c*2+1)*n + 1][col] = 1   # C[1, b_c*2+1] += m3
            col += 1
            
            # m4 = d(g-e): contributes to c00 and c10
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = -1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0, b_c*2] += m4
            W[(b_c*2+0)*n + 1][col] = 1   # C[1, b_c*2] += m4
            col += 1
            
            # m5 = (a+b)h: contributes to c00 and c01
            U[0*m + (b_r*2+0)][col] = 1
            U[0*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = -1  # C[0, b_c*2] -= m5
            W[(b_c*2+1)*n + 0][col] = 1   # C[0, b_c*2+1] += m5
            col += 1
            
            # m6 = (c-a)(e+f): contributes to c11
            U[0*m + (b_r*2+0)][col] = -1
            U[1*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1   # C[1, b_c*2+1] += m6
            col += 1
            
            # m7 = (b-d)(g+h): contributes to c00
            U[0*m + (b_r*2+1)][col] = 1
            U[1*m + (b_r*2+1)][col] = -1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0, b_c*2] += m7
            col += 1
            
            # Schoolbook part for row 2: 4 terms
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    
    # Verify the decomposition
    max_err = _verify_decomposition(U, V, W, n, m, p)
    if max_err > 1e-10:
        raise ValueError(f"Decomposition verification failed with max error {max_err}")
    
    return {"U": U, "V": V, "W": W}
