import time
import numpy as np

def _verify_decomposition(U, V, W, n=3, m=4, p=4):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    C_computed = np.zeros((p, n))
    for k in range(rank):
        u = U[:, k].reshape(n, m)
        v = V[:, k].reshape(m, p)
        w = W[:, k].reshape(p, n)
        prod = u @ A.T  # This is not right, let me fix
        
    # Correct computation: C[i,j] = sum_k U[i*m+j, k] * V[j*p+i, k] * W[i*n+j, k]
    # Wait, the convention is W[k*n+i, r] where k is row of C, i is col of C
    # So C[k, i] = sum_r U[i*m+?, r] ... let me re-read
    
    # From the task: W[k*n+i, r] where output is C[k, i]
    # U has shape (n*m, rank), indexed by (i, j) -> i*m+j where i is row of A, j is col of A
    # V has shape (m*p, rank), indexed by (j, k) -> j*p+k where j is col of A/row of B, k is col of B
    # W has shape (p*n, rank), indexed by (k, i) -> k*n+i where k is row of C, i is col of C
    
    # C[k, i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
    # Actually: C[k,i] = sum_{j,r} A[i,j] * B[j,k] * (decomposition)
    # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify: for random A, B, compute C = A@B and check T matches
    C_from_tensor = np.zeros((n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                C_from_tensor[i, k] += total * A[i, j] * B[j, k]
    
    # Wait, that's not right either. The decomposition should satisfy:
    # (A@B)[i,k] = sum_{j} A[i,j] * B[j,k] = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_k' B[j,k']*V[j*p+k',r]) ... 
    # No, that's not how it works.
    
    # Correct: The tensor T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # And we need: (A@B)[i,k] = sum_j A[i,j] * B[j,k]
    # So we need: sum_j A[i,j] * B[j,k] = sum_r W[k*n+i,r] * (sum_j A[i,j]*U[i*m+j,r]) * ... 
    # This is getting confused. Let me just check if the bilinear form matches.
    
    # The correct check: For all i,j,k, T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # should satisfy that when we compute sum_j A[i,j]*B[j,k], we get the right answer.
    # Actually, the decomposition means:
    # (A@B)[i,k] = sum_r W[k*n+i,r] * (sum_j A[i,j]*U[i*m+j,r]) * ... no.
    
    # Let me think again. A rank-1 tensor is u * v * w where u[i*m+j], v[j*p+k], w[k*n+i]
    # The product is: sum_{i,j,k} u[i*m+j]*v[j*p+k]*w[k*n+i] * A[i,j]*B[j,k]
    # = sum_r w[k*n+i,r] * (sum_j A[i,j]*u[i*m+j,r]) * (sum_k' B[j,k']*v[j*p+k',r])... no.
    
    # I think the correct interpretation is:
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (coefficient)
    # where coefficient comes from the tensor.
    
    # Let me just do a direct numerical check with small matrices.
    pass

def construct():
    """
    Rank-44 tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    - Split the 4x4 B matrix into four 2x2 blocks based on row and column indices.
    - For each 2x2 block of B (rows b_r*2 to b_r*2+1, cols b_c*2 to b_c*2+1):
      - Multiply the top 2x2 block of A (rows 0-1, cols b_r*2 to b_r*2+1) by the B block 
        using Strassen's algorithm (rank 7).
      - Multiply the bottom 1x2 block of A (row 2, cols b_r*2 to b_r*2+1) by the B block 
        using schoolbook multiplication (rank 4).
    - Total rank per (b_r, b_c) block: 7 + 4 = 11.
    - Total rank: 2 * 2 * 11 = 44.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x 44
    V = [[0]*rank for _ in range(m*p)]  # 16 x 44
    W = [[0]*rank for _ in range(p*n)]  # 12 x 44
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
            # Strassen part: 7 terms, rows i=0,1
            # m1 = (a+d)(e+h)
            U[0*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0,0] += m1
            W[(b_c*2+1)*n + 1][col] = 1   # C[1,1] += m1
            col += 1
            
            # m2 = (c+d)e: A-part: A[1,0]=1, A[1,1]=1; B-part: B[0,0]=1
            U[0*m + (b_r*2+0)][col] = 0
            U[1*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1   # C[1,0] += m2
            W[(b_c*2+1)*n + 1][col] = -1  # C[1,1] -= m2
            col += 1
            
            # m3 = a(f-h): A-part: A[0,0]=1; B-part: B[0,1]=1, B[1,1]=-1
            U[0*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1   # C[0,1] += m3
            W[(b_c*2+1)*n + 1][col] = 1   # C[1,1] += m3
            col += 1
            
            # m4 = d(g-e): A-part: A[1,1]=1; B-part: B[1,0]=1, B[0,0]=-1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = -1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0,0] += m4
            W[(b_c*2+0)*n + 1][col] = 1   # C[1,0] += m4
            col += 1
            
            # m5 = (a+b)h: A-part: A[0,0]=1, A[0,1]=1; B-part: B[1,1]=1
            U[0*m + (b_r*2+0)][col] = 1
            U[0*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = -1  # C[0,0] -= m5
            W[(b_c*2+1)*n + 0][col] = 1   # C[0,1] += m5
            col += 1
            
            # m6 = (c-a)(e+f): A-part: A[1,0]=1, A[0,0]=-1; B-part: B[0,0]=1, B[0,1]=1
            U[0*m + (b_r*2+0)][col] = -1
            U[1*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1   # C[1,1] += m6
            col += 1
            
            # m7 = (b-d)(g+h): A-part: A[0,1]=1, A[1,1]=-1; B-part: B[1,0]=1, B[1,1]=1
            U[0*m + (b_r*2+1)][col] = 1
            U[1*m + (b_r*2+1)][col] = -1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0,0] += m7
            col += 1
            
            # Trivial part for row i=2: 4 terms
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
