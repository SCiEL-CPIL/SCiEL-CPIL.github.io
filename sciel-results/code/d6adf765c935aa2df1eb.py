"""Rewritten program for matrix multiplication 2x3 by 3x4 using structured search for low-rank decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    Strategy: 
    1. Theoretical minimum rank for 2x3 * 3x4 is 12.
    2. We use a randomized search with local refinement (ALS) to find a rank-r decomposition.
    3. We initialize with random small integers/half-integers to respect the output format.
    4. We refine using alternating least squares to drive the error to 0.
    5. We verify exactness and integer/half-integer constraints.
    6. If no solution is found for a rank, we increment the rank.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Try ranks from 12 upwards
    for rank in range(12, n * m * p + 1):
        result = _find_decomposition(rank, n, m, p, trials=50)
        if result is not None:
            return result
            
    # Fallback to schoolbook (rank 24)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _find_decomposition(rank, n, m, p, trials=10):
    """
    Attempt to find a rank-r decomposition using randomized initialization and ALS refinement.
    """
    # Dimensions of factor matrices
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T[i, j, k] = 1
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Pre-compute indices for vectorization
    # T is shape (n, m, p)
    # We can reshape the equation for ALS.
    
    for trial in range(trials):
        # Initialize with small random integers/half-integers
        # Using a restricted set to help convergence to integer/half-integer solutions
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        U = np.random.choice(vals, size=(dim_u, rank))
        V = np.random.choice(vals, size=(dim_v, rank))
        W = np.random.choice(vals, size=(dim_w, rank))
        
        # Ensure non-trivial (avoid all zeros)
        if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
            continue
            
        # Refine using Alternating Least Squares (ALS)
        # We iterate: fix V, W -> solve for U; fix U, W -> solve for V; etc.
        # The objective is min || T - sum U_c V_c W_c ||_F^2
        
        # To solve for U given V, W:
        # T[i,j,k] = sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c])
        # Let M_u[i*m+j, c] = U[i*m+j, c]
        # The term for a fixed (i,j) is sum_c M_u[i*m+j, c] * Q_u[i*m+j, c] where Q_u depends on V,W
        # Actually, it's easier to vectorize.
        
        # Let's use a simple gradient descent or ALS with numpy linalg.
        # ALS step for U:
        # For each column c, U[:, c] is a vector.
        # The residual R = T - sum_{c'} U_{c'} \otimes V_{c'} \otimes W_{c'}
        # This is complex to vectorize directly for ALS without heavy memory.
        # Given small dimensions (6, 12, 8), we can afford explicit loops or small matrices.
        
        success = _refine_als(U, V, W, n, m, p, iterations=50)
        
        if success:
            # Check if the solution is close to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify exactness with rounded values
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                
    return None


def _refine_als(U, V, W, n, m, p, iterations=20):
    """
    Refine U, V, W using Alternating Least Squares to minimize reconstruction error.
    Returns True if error is below tolerance.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = U.shape[1]
    
    # T[i,j,k] = 1
    # Residual calculation:
    # R[i,j,k] = 1 - sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for _ in range(iterations):
        # Solve for U
        # For each (i,j), we have a linear system for U[i*m+j, :]
        # 1 = sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) -- wait, this sums over k too?
        # No, T[i,j,k] is for each k.
        # The equation is: For all i,j,k: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # This is not a simple linear system for U if we consider all k.
        # However, we can view the tensor as a matrix multiplication in a transformed space.
        # Let's just use Gradient Descent for simplicity and robustness on small sizes.
        
        # Gradient Descent
        lr = 0.01
        for step in range(100):
            # Compute gradients
            # dL/dU[i*m+j, c] = - sum_k (1 - sum_c' U[i*m+j, c'] V[j*p+k, c'] W[k*n+i, c']) * V[j*p+k, c] * W[k*n+i, c]
            
            # This is O(rank * n * m * p * rank) which is small.
            
            # Compute current reconstruction
            # Rec[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Instead of full GD, let's try a few ALS steps properly.
            # ALS for U:
            # Fix V, W.
            # For a fixed i, j, the vector u_ij = U[i*m+j, :] is involved in equations for all k.
            # Eq: 1 = sum_c u_ij[c] * (V[j*p+k, c] * W[k*n+i, c]) for each k.
            # Let A_k[c] = V[j*p+k, c] * W[k*n+i, c].
            # We have a system: sum_c u_ij[c] A_k[c] = 1 for all k=0..p-1.
            # This is p equations, rank variables. If rank > p, it's underdetermined.
            # We can solve min || A u - 1 ||^2.
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build matrix A of shape (p, rank)
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A u = 1 (vector of ones length p)
                    # Least squares
                    u_sol, residuals, rank_l, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = u_sol
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # Build matrix B of shape (n, rank) ? No, indices i.
                    # Eq: 1 = sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) for all i.
                    # Let B_i[c] = U[i*m+j, c] * W[k*n+i, c].
                    # System: sum_c v_jk[c] B_i[c] = 1 for all i=0..n-1.
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    v_sol, residuals, rank_l, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = v_sol
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # Eq: 1 = sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) for all j.
                    # Let C_j[c] = U[i*m+j, c] * V[j*p+k, c].
                    # System: sum_c w_ki[c] C_j[c] = 1 for all j=0..m-1.
                    C = np.zeros((m, rank))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    w_sol, residuals, rank_l, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = w_sol
            
            U, V, W = U_new, V_new, W_new
            
            # Check error
            err = _calc_error(U, V, W, n, m, p)
            if err < 1e-9:
                return True
                
    err = _calc_error(U, V, W, n, m, p)
    return err < 1e-9


def _calc_error(U, V, W, n, m, p):
    """Calculate the maximum absolute error of the decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact (within floating point epsilon) and entries are half-integers."""
    # Check half-integer constraint
    if not np.allclose(U * 2, np.round(U * 2)): return False
    if not np.allclose(V * 2, np.round(V * 2)): return False
    if not np.allclose(W * 2, np.round(W * 2)): return False
    
    # Check exactness
    err = _calc_error(U, V, W, n, m, p)
    return err < 1e-9

# EVOLVE-BLOCK-END
