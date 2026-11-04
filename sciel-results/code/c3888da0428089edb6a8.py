"""Program for 4x5 * 5x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank exact tensor decomposition for 4x5 * 5x6 multiplication.
    Returns {'U': (20, rank), 'V': (30, rank), 'W': (24, rank)}.
    """
    t0 = time.time()
    DEADLINE = 140  # seconds, safety margin from 150s limit

    n, m, p = 4, 5, 6
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24

    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that the factorization is correct."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_decomposition(rank, n, m, p, t0, deadline):
        """
        Try to find a rank-r decomposition using structured random search.
        Returns (U, V, W) if found, else None.
        """
        # We need U (n*m x r), V (m*p x r), W (p*n x r)
        # such that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Strategy: Use a structured approach where we initialize with
        # random values and then use iterative refinement.
        
        # First, let's try a simpler approach: use the fact that the tensor
        # has a specific structure. For each (i,k) pair, we need:
        # sum_j A[i,j] * B[j,k] = C[i,k]
        # In tensor form: for each (i,j,k), the contribution is 1.
        
        # Let's try a randomized approach with structured initialization.
        # We'll use the fact that for small ranks, we can try to solve
        # the system of equations directly.
        
        # Number of equations: n*m*p = 120
        # Number of unknowns: rank * (n*m + m*p + p*n) = rank * 74
        
        # For rank < 120, this is an underdetermined system, but it's nonlinear
        # (trilinear). We need a smarter approach.
        
        # Approach: Use a greedy/iterative method.
        # 1. Start with a random initialization of U, V, W
        # 2. Iteratively update one factor at a time to minimize the residual
        
        np.random.seed(42)
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - t0 > deadline:
                return None
            
            # Initialize with small random values
            U = np.random.randn(num_U, rank) * 0.5
            V = np.random.randn(num_V, rank) * 0.5
            W = np.random.randn(num_W, rank) * 0.5
            
            # Iterative refinement: alternate between updating each factor
            for iteration in range(200):
                if time.time() - t0 > deadline:
                    return None
                
                # Update U: for each (i,j), we want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is a linear system in U[i*m+j, :]
                # For fixed (i,j): sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
                # Let M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Then sum_c U[i*m+j, c] * M[k, c] = 1 for all k
                # So U[i*m+j, :] = M^{-1} * ones (if M is square and invertible)
                # But M is (p x rank), so we need rank >= p for this to work directly.
                
                # Let's use a least-squares approach:
                # For each (i,j): minimize sum_k (sum_c U[i*m+j, c] * M[k, c] - 1)^2
                # Solution: U[i*m+j, :] = (M^T M)^{-1} M^T * ones
                
                for i in range(n):
                    for j in range(m):
                        # Build M[k, c] = V[j*p+k, c] * W[k*n+i, c] for k=0..p-1, c=0..rank-1
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        # Solve least squares: M @ U[i*m+j, :] = ones(p)
                        try:
                            U[i * m + j, :], residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        except:
                            pass
                
                # Update V: for each (j,k), we want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                # For fixed (j,k): sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = 1 for all i
                # Let N[i, c] = U[i*m+j, c] * W[k*n+i, c]
                # Then sum_c V[j*p+k, c] * N[i, c] = 1 for all i
                # So V[j*p+k, :] = (N^T N)^{-1} N^T * ones
                
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, rank))
                        for i in range(n):
                            N[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        try:
                            V[j * p + k, :], residuals, rank_n, sv = np.linalg.lstsq(N, np.ones(n), rcond=None)
                        except:
                            pass
                
                # Update W: for each (k,i), we want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                # For fixed (k,i): sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for all j
                # Let O[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # Then sum_c W[k*n+i, c] * O[j, c] = 1 for all j
                # So W[k*n+i, :] = (O^T O)^{-1} O^T * ones
                
                for k in range(p):
                    for i in range(n):
                        O = np.zeros((m, rank))
                        for j in range(m):
                            O[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        try:
                            W[k * n + i, :], residuals, rank_o, sv = np.linalg.lstsq(O, np.ones(m), rcond=None)
                        except:
                            pass
            
            # Check if we found a valid solution
            if verify(U, V, W, n, m, p, tol=1e-6):
                # Convert to exact fractions if possible
                U_frac = np.zeros_like(U)
                V_frac = np.zeros_like(V)
                W_frac = np.zeros_like(W)
                
                # Try to find rational approximations
                for idx in range(num_U):
                    for c in range(rank):
                        U_frac[idx, c] = float(Fraction(U[idx, c]).limit_denominator(1000))
                for idx in range(num_V):
                    for c in range(rank):
                        V_frac[idx, c] = float(Fraction(V[idx, c]).limit_denominator(1000))
                for idx in range(num_W):
                    for c in range(rank):
                        W_frac[idx, c] = float(Fraction(W[idx, c]).limit_denominator(1000))
                
                if verify(U_frac, V_frac, W_frac, n, m, p, tol=1e-9):
                    return U_frac, V_frac, W_frac
                else:
                    # Return the floating point version if it's close enough
                    return U, V, W
        
        return None

    # Try ranks from lower to upper
    # Lower bound: max(n*m, m*p, p*n) = 30
    # But for matrix multiplication, the rank is likely higher.
    # Let's try a few ranks and see what works.
    
    # Start with schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((num_U, rank_schoolbook))
    V_school = np.zeros((num_V, rank_schoolbook))
    W_school = np.zeros((num_W, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    if not verify(U_school, V_school, W_school, n, m, p):
        # This should never happen, but just in case
        pass
    
    best_result = (U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition
    # Try ranks: 30, 40, 50, 60, 70, 80, 90, 100, 110
    for rank in [30, 40, 50, 60, 70, 80, 90, 100, 110]:
        if time.time() - t0 > DEADLINE - 10:
            break
        
        result = try_decomposition(rank, n, m, p, t0, DEADLINE - 5)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, n, m, p, tol=1e-8):
                best_result = (U, V, W)
                best_rank = rank
                break  # Found a valid decomposition at this rank
    
    U_best, V_best, W_best = best_result
    
    # Convert to lists for return
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
