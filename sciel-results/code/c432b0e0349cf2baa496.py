"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # The tensor has shape (n, m, p) and T[i,j,k] = 1 for all i,j,k
    # Actually, the matrix multiplication tensor is T[i,j,k] = 1 for all valid indices
    # because (AB)[i,k] = sum_j A[i,j] B[j,k]
    
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention.
    
    # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
    # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
    
    # The product sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k]
    # where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is equivalent to saying the tensor T (which is all ones) has a rank decomposition.
    
    # Let's think about this differently. The matrix multiplication tensor M_{i,j,k} = 1
    # We need to find rank-minimal decomposition.
    
    # Known results: For 2x3 * 3x7 matrix multiplication:
    # The trivial bound is 2*3*7 = 42
    # We can try to find better decompositions.
    
    # Let's use a randomized approach with exact verification.
    # First, let's try to find a decomposition of rank r < 42.
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check that for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
        """Fast verification using numpy."""
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        # U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank)
        # For each c, we have outer products
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Let's build the tensor and check
        T = np.zeros((n, m, p))
        for c in range(rank):
            u = U_mat[:, c].reshape(n, m)
            v = V_mat[:, c].reshape(m, p)
            w = W_mat[:, c].reshape(p, n)
            # The contribution to T[i,j,k] is u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        return np.allclose(T, np.ones((n,m,p)), atol=tol)
    
    # Try to find a lower rank decomposition
    # Start with rank = 42 (schoolbook) and try to reduce
    
    best_rank = rank_schoolbook
    best_decomp = None
    
    # Generate schoolbook decomposition
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try to find a decomposition with lower rank using random initialization and optimization
    # We'll use a simple approach: try ranks from 30 down to 42, and for each rank,
    # try random initializations and check if we can get an exact decomposition.
    
    # Actually, let's think about this more carefully. The problem is to find the tensor rank
    # of the matrix multiplication tensor for 2x3 * 3x7.
    
    # Let's try a structured approach. We can try to find a decomposition where
    # the factors have a special structure.
    
    # For now, let's try to use the fact that we can combine some of the schoolbook terms.
    
    # Let's try ranks from 35 to 42 and see if we can find valid decompositions.
    
    # Use a smarter approach: parameterize the decomposition and use numerical optimization
    # to find one that exactly satisfies the constraints.
    
    # Actually, for exact decompositions with half-integer entries, we need to be careful.
    
    # Let's try a different approach: use the known structure of matrix multiplication.
    
    # The matrix multiplication tensor for (n,m,p) has rank at most n*m*p.
    # For small cases, we can try to find the exact rank.
    
    # Let's try to find a decomposition of rank r by solving a system of equations.
    # For rank r, we have:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of n*m*p = 42 equations in r*(n*m + m*p + p*n) = r*(6+21+14) = 41r unknowns.
    # For r = 42, we have 42*41 = 1722 unknowns and 42 equations, so it's underdetermined.
    # For smaller r, it becomes more constrained.
    
    # Let's try to find a decomposition with rank 40, 38, etc.
    
    # Use a randomized approach with exact verification
    random.seed(42)
    np.random.seed(42)
    
    for target_rank in range(41, 29, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            U_r = np.random.randn(n*m, target_rank) * 0.5
            V_r = np.random.randn(m*p, target_rank) * 0.5
            W_r = np.random.randn(p*n, target_rank) * 0.5
            
            # Check if this random initialization happens to work (unlikely)
            # Instead, let's try to optimize towards a valid solution
            
            # Simple gradient-free approach: try to adjust the factors
            # to satisfy the constraints.
            
            # Actually, for this problem, let's try a more structured approach.
            # We can try to find a decomposition where some of the rank-1 terms
            # are combined.
            
            pass
    
    # If we didn't find a better decomposition, return the schoolbook one
    U_final, V_final, W_final = best_decomp
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}
# EVOLVE-BLOCK-END
