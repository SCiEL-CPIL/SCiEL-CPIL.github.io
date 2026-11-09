"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # U index: (i_off + li) * m + (j_off + lj)
        # V index: (j_off + lj) * p + (k_off + lk)
        # W index: (k_off + lk) * n + (i_off + li)
        
        # M1 = (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # M3 = a(f-h)
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    # 4 Strassen blocks for the 2x4 x 4x4 part (rows 0-1 of A)
    add_strassen_222(0, 0, 0)  # A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_222(0, 2, 0)  # A[0:2,2:4] x B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_222(0, 0, 2)  # A[0:2,0:2] x B[0:2,2:4] -> C[0:2,2:4]
    add_strassen_222(0, 2, 2)  # A[0:2,2:4] x B[2:4,2:4] -> C[0:2,2:4]
    
    # 4 rank-4 blocks for row 2 of A
    add_122(2, 0, 0)  # A[2,0:2] x B[0:2,0:2] -> C[2,0:2]
    add_122(2, 2, 0)  # A[2,2:4] x B[2:4,0:2] -> C[2,0:2]
    add_122(2, 0, 2)  # A[2,0:2] x B[0:2,2:4] -> C[2,2:4]
    add_122(2, 2, 2)  # A[2,2:4] x B[2:4,2:4] -> C[2,2:4]
    
    # Verify correctness
    if time.time() < deadline:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Test with random matrices
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m))
        B_test = np.random.randint(-5, 6, size=(m, p))
        C_expected = A_test @ B_test
        
        # Compute via decomposition: C[k*n+i, :] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # More precisely: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (decomposition)
        # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta condition
        # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        # And we need: sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
        # Wait, that's not right. The tensor decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 if this is a valid (i,j,k) triple for matrix mult
        # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor is T[i,j,k] = 1 for all valid (i,j,k)
        # And we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        # And 0 otherwise (but all i,j,k are valid here)
        
        # Let's verify: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
        # This is because C[i,k] = sum_j A[i,j]*B[j,k], and the decomposition recovers this
        # via: sum_r (sum_j A[i,j]*U[i*m+j,r]) * ... no that's wrong.
        
        # Correct verification: The tensor T_{ijk} for matrix mult M_{n,m,p} is:
        # T_{ijk} = 1 if the (i,j,k) index is valid (always is for full ranges)
        # The rank-r decomposition: T_{ijk} = sum_{r=1}^r U_{ij,r} * V_{jk,r} * W_{ki,r}
        # (with appropriate index mappings)
        
        # So check: for all i in [0,n), j in [0,m), k in [0,p):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    max_err = max(max_err, abs(val - 1))
        
        if max_err > 1e-9:
            print(f"Verification failed with max error {max_err}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
