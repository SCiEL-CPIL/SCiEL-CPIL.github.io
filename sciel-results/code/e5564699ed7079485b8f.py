"""Combined program: Winograd-Fourier exact decomposition for 3x5x6 matrix multiplication."""
import time
import itertools
import random
from fractions import Fraction
from math import gcd

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _verify_full(U, V, W, n, m, p):
    """Full verification using Fraction for exactness."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s deadline, 15s safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    rank_schoolbook = n * m * p  # 90
    
    # Winograd Fourier Transform approach for 3x5x6 matrix multiplication
    # Decompose m=5 into frequency domain using 5-point DFT
    # Decompose n=3 and p=6 using 3-point DFT
    
    # The key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # Using DFT along the j dimension (length 5):
    # Let A_hat[i, omega] = sum_j A[i,j] * exp(-2*pi*i*j*omega/5)
    # Let B_hat[omega, k] = sum_j B[j,k] * exp(2*pi*i*j*omega/5)
    # Then C[i,k] = (1/5) * sum_omega A_hat[i,omega] * B_hat[omega,k]
    
    # For each omega in {0,1,2,3,4}, we need to multiply A_hat[i,omega] (3x1) by B_hat[omega,k] (1x6)
    # This is a 3x1 times 1x6 multiplication, which has rank 3 (schoolbook)
    # Total rank = 5 * 3 = 15... but we need to be more careful.
    
    # Actually, the DFT introduces complex coefficients. Let's use a different approach.
    
    # Winograd algorithm for convolution-like structure:
    # For n=3, m=5, p=6, we can use the fact that:
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    # Let's try a direct algebraic approach with rank minimization.
    # Start with schoolbook rank 90 and try to reduce.
    
    # Build schoolbook decomposition first as baseline
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Now try to reduce rank using algebraic search
    import numpy as np
    
    def try_rank_reduction(U_init, V_init, W_init, n, m, p, max_time=120):
        """
        Try to find a lower-rank exact decomposition using alternating least squares.
        """
        R = len(U_init[0])
        best_U, best_V, best_W = U_init, V_init, W_init
        best_rank = R
        
        nm, mp, pn = n*m, m*p, p*n
        
        # Convert to numpy
        U_np = np.array(U_init, dtype=float)
        V_np = np.array(V_init, dtype=float)
        W_np = np.array(W_init, dtype=float)
        
        def solve_for_U(V_fixed, W_fixed, r, n, m, p):
            """Given V and W of rank r, solve for U such that the tensor matches."""
            U_sol = np.zeros((n*m, r))
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V_fixed[j*p+k, :] * W_fixed[k*n+i, :]
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_sol[i*m+j, :] = x
                    except:
                        pass
            return U_sol
        
        def solve_for_V(U_fixed, W_fixed, r, m, p, n):
            """Given U and W of rank r, solve for V."""
            V_sol = np.zeros((m*p, r))
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U_fixed[i*m+j, :] * W_fixed[k*n+i, :]
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_sol[j*p+k, :] = x
                    except:
                        pass
            return V_sol
        
        def solve_for_W(U_fixed, V_fixed, r, p, n, m):
            """Given U and V of rank r, solve for W."""
            W_sol = np.zeros((p*n, r))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U_fixed[i*m+j, :] * V_fixed[j*p+k, :]
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_sol[k*n+i, :] = x
                    except:
                        pass
            return W_sol
        
        def check_exact(U_c, V_c, W_c, n, m, p, tol=1e-9):
            """Check if the decomposition is exact (within tolerance)."""
            r = len(U_c[0])
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for s_idx in range(r):
                            s += U_c[i*m+j][s_idx] * V_c[j*p+k][s_idx] * W_c[k*n+i][s_idx]
                        if abs(s - 1.0) > tol:
                            return False
            return True
        
        def check_exact_frac(U_c, V_c, W_c, n, m, p):
            """Check exactness using Fraction."""
            r = len(U_c[0])
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for s_idx in range(r):
                            s += Fraction(U_c[i*m+j][s_idx]) * Fraction(V_c[j*p+k][s_idx]) * Fraction(W_c[k*n+i][s_idx])
                        if s != 1:
                            return False
            return True
        
        # Try to reduce rank from best_rank down to some minimum
        min_rank_to_try = 15  # Lower bound estimate
        
        random.seed(42)
        
        for target_rank in range(best_rank - 1, min_rank_to_try - 1, -1):
            if time.time() > deadline - 10:
                break
            
            found = False
            num_restarts = 3
            
            for restart in range(num_restarts):
                if time.time() > deadline - 10:
                    break
                
                # Random initialization
                U_r = np.random.randn(n*m, target_rank) * 0.1
                V_r = np.random.randn(m*p, target_rank) * 0.1
                W_r = np.random.randn(p*n, target_rank) * 0.1
                
                # Alternating least squares
                for iteration in range(100):
                    if time.time() > deadline - 10:
                        break
                    
                    U_r = solve_for_U(V_r, W_r, target_rank, n, m, p)
                    V_r = solve_for_V(U_r, W_r, target_rank, m, p, n)
                    W_r = solve_for_W(U_r, V_r, target_rank, p, n, m)
                
                # Check if we found an exact solution
                if check_exact(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-8):
                    if check_exact_frac(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        best_U = U_r.tolist()
                        best_V = V_r.tolist()
                        best_W = W_r.tolist()
                        best_rank = target_rank
                        found = True
                        break
            if found:
                continue
            else:
                if time.time() > deadline - 15:
                    break
        
        return best_U, best_V, best_W, best_rank
    
    # Try to reduce rank
    try:
        best_U, best_V, best_W, best_rank = try_rank_reduction(
            best_U, best_V, best_W, n, m, p
        )
    except Exception as e:
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
