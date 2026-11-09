"""Tensor decomposition for matrix multiplication 3x3 * 3x8."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[:,c] V^T[:,c] W^T[:,c] = T where T[i*m+j, j*p+k, k*n+i] = 1."""
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    U_3d = U.reshape(n * m, rank)
    V_3d = V.reshape(m * p, rank)
    W_3d = W.reshape(p * n, rank)
    
    T_comp = np.einsum('ar,br,cr->abc', U_3d, V_3d, W_3d)
    
    return np.allclose(T_comp, T, atol=tol)

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) with shapes (4,7), (4,7), (4,7).
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Column indices for each product
    # Product 1: M1
    U[0, 0] = 1  # a11
    U[3, 0] = 1  # a22
    V[0, 0] = 1  # b11
    V[3, 0] = 1  # b22
    W[0, 0] = 1  # c11
    W[3, 0] = 1  # c22
    
    # Product 2: M2
    U[2, 1] = 1  # a21
    U[3, 1] = 1  # a22
    V[0, 1] = 1  # b11
    W[2, 1] = 1  # c21
    W[3, 1] = -1 # c22
    
    # Product 3: M3
    U[0, 2] = 1  # a11
    V[1, 2] = 1  # b12
    V[3, 2] = -1 # b22
    W[1, 2] = 1  # c12
    W[3, 2] = 1  # c22
    
    # Product 4: M4
    U[3, 3] = 1  # a22
    V[2, 3] = 1  # b21
    V[0, 3] = -1 # b11
    W[0, 3] = 1  # c11
    W[2, 3] = 1  # c21
    
    # Product 5: M5
    U[0, 4] = 1  # a11
    U[1, 4] = 1  # a12
    V[3, 4] = 1  # b22
    W[0, 4] = -1 # c11
    W[1, 4] = 1  # c12
    
    # Product 6: M6
    U[2, 5] = 1  # a21
    U[0, 5] = -1 # a11
    V[0, 5] = 1  # b11
    V[1, 5] = 1  # b12
    W[3, 5] = 1  # c22
    
    # Product 7: M7
    U[1, 6] = 1  # a12
    U[3, 6] = -1 # a22
    V[2, 6] = 1  # b21
    V[3, 6] = 1  # b22
    W[0, 6] = 1  # c11
    
    return U, V, W

def _build_strassen_3x3_3x8(n, m, p, deadline):
    """
    Try to construct a low-rank decomposition for 3x3 * 3x8 using Strassen blocks.
    
    Strategy: Decompose the 3x3 matrix A into overlapping 2x2 blocks and use
    Strassen's algorithm on each pair, then combine results.
    
    For 3x3 * 3x8, we can think of it as:
    - Split A (3x3) into overlapping 2x2 blocks
    - Split B (3x8) into overlapping 2x8 blocks
    - Use Strassen on each 2x2 * 2x2 subproblem
    
    A more systematic approach: use the fact that 3x3 * 3x8 can be decomposed
    into a sum of 2x2 * 2x8 blocks using Strassen.
    
    Let's try a direct construction using Kronecker products and Strassen.
    """
    start_time = time.time()
    
    # Get Strassen 2x2 decomposition
    S_U, S_V, S_W = _strassen_2x2_decomposition()
    
    # For 3x3 * 3x8, we can partition as follows:
    # Think of 3x3 matrix A as having entries A[i,j] for i,j in {0,1,2}
    # Think of 3x8 matrix B as having entries B[j,k] for j in {0,1,2}, k in {0,...,7}
    # Output C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can decompose this using 2x2 Strassen blocks:
    # Block 1: A[0:2, 0:2] * B[0:2, :] -> contributes to C[0:2, :]
    # Block 2: A[1:3, 1:3] * B[1:3, :] -> contributes to C[1:3, :]
    # But this doesn't cover all entries cleanly.
    #
    # Alternative: Use a direct tensor decomposition approach.
    # The tensor T for 3x3*3x8 has dimension 9*24*24 = 5184.
    # Schoolbook rank is 9*24 = 216.
    #
    # Let's try to find a rank < 216 decomposition using optimization.
    
    # First, try a structured approach: partition into 2x2 blocks
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    # 
    # We can write C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # For each pair (i,k), this is a bilinear form.
    # The total number of such forms is 3*8 = 24.
    # Each involves a sum over j=0,1,2.
    # 
    # Total schoolbook terms: 3*3*8 = 72.
    # 
    # To reduce rank, we need to share computations across different (i,k) pairs.
    # 
    # Let's try: group columns of B and use Strassen on 2x2 blocks of A.
    # 
    # Actually, let's try a simpler approach: use the fact that we can compute
    # 2x2 * 2x8 using 7*8 = 56 products instead of 2*2*8 = 32 products.
    # Wait, that's worse! Strassen helps when output is small relative to inputs.
    # 
    # For 2x2 * 2x8: schoolbook = 2*2*8 = 32, Strassen = 7*8 = 56. Worse!
    # 
    # For 2x2 * 2x2: schoolbook = 4, Strassen = 7. Also worse!
    # Wait, I made an error. Strassen for 2x2*2x2 uses 7 multiplications vs 4 schoolbook.
    # So Strassen is worse for 2x2!
    # 
    # Strassen helps for larger matrices. For 4x4*4x4, Strassen uses 7*4^2 = 112
    # vs schoolbook 4*4*4 = 64. Still worse!
    # 
    # Actually, Strassen is asymptotically better but has higher constants.
    # For small matrices, schoolbook is often better.
    # 
    # So we should focus on finding a different low-rank decomposition.
    # 
    # Let's try optimization-based approach.
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a low-rank approximation
    # Start with schoolbook rank and try to reduce
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try ranks from lower to higher
    for rank in range(50, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Try to find a rank-r decomposition
        # Use random initialization and gradient descent
        for trial in range(10):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + trial)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Optimize to minimize ||T - T_comp||_F
            # T_comp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            
            # Use a simple iterative approach
            for step in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Compute current tensor
                T_comp = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T - T_comp
                loss = np.sum(residual ** 2)
                
                if loss < 1e-12:
                    break
                
                # Compute gradients
                # d/dU[a,r] = sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
                grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
                grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
                grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Simple gradient descent
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((T - T_comp) ** 2)
            
            if loss < 1e-9:
                # Found a valid decomposition
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
        else:
            continue
        break
    
    if best_U is not None:
        return best_U, best_V, best_W
    
    # If optimization didn't find anything, return None
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition first
    better = _build_strassen_3x3_3x8(n, m, p, deadline)
    if better is not None:
        U, V, W = better
        if _verify(U, V, W, n, m, p):
            # Convert to lists of lists with integer/half-integer entries
            U_list = [[int(round(x * 2)) / 2 for x in row] for row in U.tolist()]
            V_list = [[int(round(x * 2)) / 2 for x in row] for row in V.tolist()]
            W_list = [[int(round(x * 2)) / 2 for x in row] for row in W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    U_list = [[int(round(x * 2)) / 2 for x in row] for row in U.tolist()]
    V_list = [[int(round(x * 2)) / 2 for x in row] for row in V.tolist()]
    W_list = [[int(round(x * 2)) / 2 for x in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
