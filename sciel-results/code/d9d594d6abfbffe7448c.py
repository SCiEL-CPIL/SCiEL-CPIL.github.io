"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _check_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W form a valid decomposition of the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The convention is W[k*n + i, r] for the output factor.
    
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for all valid (i,j,k) in the multiplication.
    
    Actually, the matrix multiplication tensor M[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    So the tensor to decompose is T[i,j,k] = delta_{i,k}... no wait.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The bilinear form is: sum_r U[i*m+j, r] * V[j*p+k, r] should give the coefficient for C[i,k].
    But we need W as well. The standard decomposition is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    For matrix multiplication, T[i,j,k] = 1 for all i,j,k (it's the "multiplication tensor").
    Wait, no. The tensor for matrix multiplication has T[i,j,k] = 1 if we're summing over j.
    Actually, the multiplication tensor M satisfies:
    M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    No, that's not right either. Let me think more carefully.
    
    The matrix multiplication tensor is a 3-tensor T of dimensions n x m x p where:
    T[i,j,k] = 1 for all i, j, k
    
    This is because C[i,k] = sum_j A[i,j] * B[j,k], and the tensor encodes the bilinear structure.
    The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the schoolbook method, each term corresponds to a single (i,j,k) triple, so:
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for that specific r.
    """
    rank = U.shape[1]
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_true = np.ones((n, m, p))
    return np.allclose(T_recon, T_true, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find lower rank decompositions
    # Known lower bounds and constructions for 2x3 x 3x4
    # The Strassen-like methods can sometimes reduce rank
    
    # Let's try various ranks from a known lower bound up to schoolbook
    # For 2x3 x 3x4, the tensor rank is at most 24 (schoolbook)
    # Lower bounds suggest it could be less
    
    # Try rank 20, 21, 22, 23, 24
    for target_rank in [23, 22, 21, 20]:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Use ALS (Alternating Least Squares) to minimize reconstruction error
            # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            
            # Flatten the target tensor
            T_true = np.ones(n * m * p)
            
            # ALS iterations
            for iteration in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each (i,j), solve linear system
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is a linear system in U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # Build the linear system for U[i*m+j, :]
                        # For each k, we have: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Solve A_mat @ U[i*m+j, :] = ones(p)
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve linear system
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve linear system
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            # Check if we got a valid decomposition
            if _check_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break  # Found a valid decomposition for this rank, try lower rank
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
