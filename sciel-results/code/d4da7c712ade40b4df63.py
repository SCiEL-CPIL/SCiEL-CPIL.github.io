"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    def add_strassen(U, V, W, col, i0, j0, k0):
        """Add one Strassen 2x2x2 block. Returns col+7."""
        # M1 = (A11+A22)(B11+B22)
        U[i0*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
        V[j0*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M2 = (A21+A22)B11
        U[(i0+1)*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+(i0+1), col] += 1; W[(k0+1)*n+(i0+1), col] -= 1
        col += 1
        # M3 = A11(B12-B22)
        U[i0*m+j0, col] += 1
        V[j0*p+(k0+1), col] += 1; V[(j0+1)*p+(k0+1), col] -= 1
        W[(k0+1)*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M4 = A22(B21-B11)
        U[(i0+1)*m+(j0+1), col] += 1
        V[(j0+1)*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+(i0+1), col] += 1; W[k0*n+i0, col] += 1
        col += 1
        # M5 = (A11+A12)B22
        U[i0*m+j0, col] += 1; U[i0*m+(j0+1), col] += 1
        V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] -= 1; W[(k0+1)*n+i0, col] += 1
        col += 1
        # M6 = (A21-A11)(B11+B12)
        U[(i0+1)*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+(k0+1), col] += 1
        W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M7 = (A12-A22)(B21+B22)
        U[i0*m+(j0+1), col] += 1; U[(i0+1)*m+(j0+1), col] -= 1
        V[(j0+1)*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            u_vec = U_arr[:, r]
            v_vec = V_arr[:, r]
            w_vec = W_arr[:, r]
            
            # u_vec is length n*m, reshape to (n, m)
            u_mat = u_vec.reshape(n, m)
            # v_vec is length m*p, reshape to (m, p)
            v_mat = v_vec.reshape(m, p)
            # w_vec is length p*n, reshape to (p, n) using trace convention W[k*n+i]
            w_mat = w_vec.reshape(p, n).T  # now (n, p)
            
            C_decomposed += u_mat @ v_mat * w_mat
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            return False
        return True
    
    # Build rank-50 decomposition: 6 Strassen blocks + 8 direct terms
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # 4 Strassen blocks for columns 0-3 (covering j=0,1 and j=2,3)
    col = add_strassen(U, V, W, col, 0, 0, 0)
    col = add_strassen(U, V, W, col, 0, 0, 2)
    col = add_strassen(U, V, W, col, 0, 2, 0)
    col = add_strassen(U, V, W, col, 0, 2, 2)
    
    # 2 Strassen blocks for columns 4-5
    col = add_strassen(U, V, W, col, 0, 0, 4)
    col = add_strassen(U, V, W, col, 0, 2, 4)
    
    # 8 direct terms for column 6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank, f"col={col} != rank={rank}"
    
    # Verify
    if not verify_decomposition(U, V, W, rank):
        # Fallback: schoolbook decomposition rank=56
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        assert col == rank
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
