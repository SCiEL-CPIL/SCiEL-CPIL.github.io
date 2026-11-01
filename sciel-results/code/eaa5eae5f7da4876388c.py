"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try to find a low-rank exact decomposition
    for target_rank in [14, 15, 16, 17, 18]:
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, target_rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook rank n*m*p
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

def _try_rank(n, m, p, rank):
    """Try to find an exact rank decomposition."""
    import numpy as np
    import time
    
    start = time.time()
    deadline = start + 130
    
    # For rank 14, use a structured construction
    if rank == 14:
        result = _construct_rank14(n, m, p)
        if result is not None:
            return result
    
    # For other ranks, use schoolbook if rank >= n*m*p
    if rank >= n * m * p:
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
    
    return None

def _construct_rank14(n, m, p):
    """
    Construct a rank-14 decomposition for (2,3,3) matrix multiplication.
    
    Strategy: Use the fact that the tensor has a product-like structure.
    We can write the (2,3,3) multiplication as two independent (1,3,3) multiplications.
    
    For a (1,3,3) multiplication: y_k = sum_j A_j * B_{j,k}
    We can use a rank-7 decomposition for each (1,3,3), giving total rank 14.
    
    The rank-7 for (1,3,3) comes from the fact that we can compute 3 dot products
    of length 3 using 7 multiplications via a structured approach.
    """
    # Known: R(1,3,3) = 7
    # We'll construct it explicitly.
    # 
    # For computing y_k = A_0*B_{0,k} + A_1*B_{1,k} + A_2*B_{2,k} for k=0,1,2
    # We use 7 intermediate products.
    #
    # One construction:
    # Let's use the following approach based on "winograd" style:
    # 
    # Actually, let me use a direct algebraic construction.
    # The key insight: we can share some products between the three outputs.
    #
    # Define 7 products:
    # P0 = A_0 * B_{0,0}
    # P1 = A_1 * B_{1,0}  
    # P2 = A_2 * B_{2,0}
    # P3 = A_0 * B_{0,1}
    # P4 = A_1 * B_{1,1}
    # P5 = A_2 * B_{2,1}
    # P6 = (A_0+A_1+A_2) * (B_{0,2}+B_{1,2}+B_{2,2})
    #
    # But this doesn't directly give us the outputs. Let me think differently.
    #
    # Alternative: Use the fact that R(1,3,3) = 7 is a known result.
    # Let me construct it using a different method.
    #
    # Actually, for (1,3,3), we have 9 input entries (3 for A, 9 for B) and 3 outputs.
    # The tensor is of size 3x3x3 = 27, but with the first index being 1, it's 3x3 = 9 entries.
    # Wait, I'm confusing myself.
    #
    # Let me restart with a cleaner approach.
    #
    # For the full (2,3,3) problem:
    # Input A is 2x3, input B is 3x3, output C is 2x3.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We need to express this as sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    #
    # Let me try a direct construction with rank 14.
    # I'll use a parameterization and verify.
    
    # Based on known results, R(2,3,3) = 14.
    # Let me try to find an explicit construction.
    
    # Approach: Use the "baby-step giant-step" or structured embedding.
    # 
    # Consider embedding the 3x3 multiplication into a larger problem.
    # For a 3x3 matrix multiplication, we know R(3,3,3) >= 19.
    # But here we have a 2x3 times 3x3, which is different.
    
    # Let me try a different tactic: use random initialization with ALS and then
    # try to find an exact rational solution.
    
    np.random.seed(12345)
    
    # Initialize with random values
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Define the target tensor
    T = np.ones((n, m, p))
    
    # ALS iterations to find an approximate solution
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                # For fixed (i,j): T[i,j,:] = U[i*m+j,:] @ (V[j,:] * W[:,i]).T
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve U_row @ A.T = T[i,j,:]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A.T, T[i, j, :].reshape(-1, 1), rcond=None)[0].flatten()
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                # For fixed (j,k): T[:,j,k] = V[j*p+k,:] @ (U[:,j,:] * W[k*n,:]).T
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(B.T, T[:, j, k].reshape(-1, 1), rcond=None)[0].flatten()
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                # For fixed (k,i): T[i,:,k] = W[k*n+i,:] @ (U[i*m,:,:] * V[:,p+k,:]).T
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(C.T, T[i, :, k].reshape(-1, 1), rcond=None)[0].flatten()
                except:
                    pass
        
        # Check convergence
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    err += (s - 1) ** 2
        if err < 1e-10:
            break
    
    # Check if we got close enough
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                max_err = max(max_err, abs(s - 1))
    
    if max_err < 1e-6:
        # Try to round to rationals
        def to_fraction_array(arr, max_denom=1000):
            frac_arr = []
            for val in arr.flatten():
                best_frac = Fraction(val).limit_denominator(max_denom)
                frac_arr.append(best_frac)
            return frac_arr
        
        U_frac = to_fraction_array(U)
        V_frac = to_fraction_array(V)
        W_frac = to_fraction_array(W)
        
        # Verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for t in range(rank):
                        s += U_frac[i * m * rank + j * rank + t] * V_frac[j * p * rank + k * rank + t] * W_frac[k * n * rank + i * rank + t]
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            U_list = [[int(f) if f.denominator == 1 else float(f) for f in U_frac[i * rank:(i + 1) * rank]] for i in range(n * m)]
            V_list = [[int(f) if f.denominator == 1 else float(f) for f in V_frac[j * rank:(j + 1) * rank]] for j in range(m * p)]
            W_list = [[int(f) if f.denominator == 1 else float(f) for f in W_frac[k * rank:(k + 1) * rank]] for k in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
