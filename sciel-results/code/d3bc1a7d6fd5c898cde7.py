"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randint(-5, 5, size=(n, m)).astype(float)
    B = np.random.randint(-5, 5, size=(m, p)).astype(float)
    C_expected = A @ B
    
    # Compute using decomposition
    C_computed = np.zeros((n, p), dtype=float)
    for t in range(rank):
        u_vec = U_arr[:, t]
        v_vec = V_arr[:, t]
        w_vec = W_arr[:, t]
        
        # u_vec is length n*m, reshape to (n,m)
        u_mat = u_vec.reshape((n, m))
        v_mat = v_vec.reshape((m, p))
        w_mat = w_vec.reshape((p, n))
        
        # The term is: w_mat * (u_mat * A) * (B * v_mat) -- no, that's not right
        # Actually: sum_t W_t[i,k] * (sum_{i',j'} U_t[i',j'] * A[i',j']) * (sum_{j',k'} V_t[j',k'] * B[j',k'])
        # But that's not how tensor decomposition works for matmul.
        
        # For matmul tensor: T[i,j,k] = 1 if term is A[i,j]*B[j,k] -> C[i,k]
        # Decomposition: C[i,k] = sum_t W_t[i,k] * U_t[i,j] * V_t[j,k]
        # Wait, that's not right either. Let me think again.
        
        # The tensor for matmul A(n,m) * B(m,p) -> C(n,p) is:
        # T_{i,j,k} where the term is A[i,j] * B[j,k] contributing to C[i,k]
        # But j is shared between A and B.
        
        # The standard tensor decomposition:
        # C[i,k] = sum_t sum_{j} U_t[i,j] * V_t[j,k] * W_t[i,k] * A[i,j] * B[j,k]
        # No, that's not right.
        
        # Correct formulation:
        # C[i,k] = sum_t W_t[i,k] * (sum_j U_t[i,j] * A[i,j]) * (sum_j V_t[j,k] * B[j,k])
        # This is NOT correct because the sums over j are independent.
        
        # The correct tensor decomposition for matmul:
        # C[i,k] = sum_t (sum_j U_t[i,j] * A[i,j]) * (sum_j V_t[j,k] * B[j,k]) * W_t[i,k]
        # But this would require U and V to depend on i,k too, which they don't in standard form.
        
        # Actually, the standard form is:
        # C[i,k] = sum_t W_t[i,k] * X_t * Y_t
        # where X_t = sum_{i',j'} U_t[i',j'] * A[i',j']
        # and Y_t = sum_{j',k'} V_t[j',k'] * B[j',k']
        
        # This is a rank-t decomposition of the matmul tensor.
        
        x_val = np.sum(U_arr[:, t] * A.flatten())
        y_val = np.sum(V_arr[:, t] * B.flatten())
        C_computed += W_arr[:, t].reshape((p, n)).T * x_val * y_val
    
    # Check
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Schoolbook decomposition: rank = n*m*p = 72
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

# EVOLVE-BLOCK-END
