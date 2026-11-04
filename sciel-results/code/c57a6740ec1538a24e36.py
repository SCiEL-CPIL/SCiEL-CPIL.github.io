"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses structured algebraic construction and randomized exact search with verification.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    def schoolbook_decomposition():
        rank_schoolbook = n * m * p
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = schoolbook_decomposition()
    
    def verify_half_integer(U, V, W):
        """
        Verify that U, V, W with half-integer entries exactly decompose the tensor.
        """
        U_int = (2 * U).astype(int)
        V_int = (2 * V).astype(int)
        W_int = (2 * W).astype(int)
        
        if not np.allclose(U * 2, U_int) or not np.allclose(V * 2, V_int) or not np.allclose(W * 2, W_int):
            return False
            
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                    if val != 8:
                        return False
        return True
    
    def try_structured_decomposition(R):
        """
        Try to find a rank-R decomposition using a structured approach.
        We use the fact that the tensor is all-ones and try to decompose it
        using a combination of algebraic identities.
        """
        # Strategy: use random half-integer vectors and solve a linear system
        # For each rank-1 tensor (u_r, v_r, w_r), the contribution to T[i,j,k] is
        # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        # We need sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all i,j,k
        
        # This is a system of 40 equations in R unknowns (if we fix the structure)
        # But the equations are nonlinear in the entries.
        
        # Alternative: fix U and V, then W is determined by a linear system.
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # This is linear in W if U and V are fixed.
        
        # So: fix U and V (random half-integer), then solve for W.
        # The system is: for each (i,j,k), sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
        # This is a linear system in the entries of W.
        
        # Number of equations: n*m*p = 40
        # Number of unknowns: p*n*R = 10*R
        # So we need R >= 4 to have enough unknowns.
        
        # Let's try: for a given R, generate random U and V with small half-integer entries,
        # then solve the linear system for W, and check if W has half-integer entries.
        
        for attempt in range(200):
            if time.time() > deadline - 10:
                return None
            
            # Generate random U and V with small half-integer entries
            # Use values in {-1, -0.5, 0, 0.5, 1} for U and V
            U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, R))
            V_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, R))
            
            # Build the linear system for W
            # Unknowns: W[k*n+i, r] for k in range(p), i in range(n), r in range(R)
            # Total unknowns: p*n*R = 10*R
            # Equations: for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            num_unknowns = p * n * R
            num_equations = n * m * p
            
            A_mat = np.zeros((num_equations, num_unknowns))
            b_vec = np.ones(num_equations)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            # W[k*n+i, r] is at index (k*n+i)*R + r
                            w_idx = (k*n+i)*R + r
                            A_mat[eq_idx, w_idx] = U_rand[i*m+j, r] * V_rand[j*p+k, r]
                        eq_idx += 1
            
            # Solve the linear system
            try:
                # Use least squares if overdetermined
                W_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            # Check if the solution is close to half-integers
            W_half = np.round(W_flat * 2) / 2
            
            # Verify the solution
            W_mat = W_half.reshape(p*n, R)
            
            # Check if W_mat satisfies the equations
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_mat[k*n+i, r]
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-9:
                return U_rand, V_rand, W_mat
        
        return None
    
    best_rank = n * m * p
    best_decomp = (U_school, V_school, W_school)
    
    # Try ranks from lower bound up to schoolbook
    candidate_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    
    for target_rank in candidate_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 15:
            break
            
        result = try_structured_decomposition(target_rank)
        if result is not None:
            U_found, V_found, W_found = result
            if verify_half_integer(U_found, V_found, W_found):
                best_rank = target_rank
                best_decomp = (U_found, V_found, W_found)
                break
    
    if best_rank == n * m * p:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        U_best, V_best, W_best = best_decomp
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
