"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i1 in range(n):
        for r in range(m):
            for j2 in range(p):
                i = i1 * m + r
                j = r * p + j2
                k = i1 * n + j2
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-6):
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.max(np.abs(recon - T))
    return err < tol

def _is_half_integer(mat, tol=1e-6):
    return np.allclose(2 * mat, np.round(2 * mat), atol=tol)

def _build_rank18():
    """Construct the known rank-18 decomposition (two Strassen blocks + schoolbook for col 4)."""
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _try_rank17_structured():
    """
    Try to find a rank-17 decomposition by sharing one term between the two Strassen blocks.
    
    The key idea: In the rank-18 solution, we have two independent Strassen blocks (cols 0-1, 2-3)
    plus 4 schoolbook terms for col 4. To get rank 17, we need to save one term.
    
    Strategy: Try to express one of the Strassen terms from block 1 as a combination of
    terms from block 2, or vice versa, by adjusting the W coefficients.
    
    More specifically: The Strassen algorithm for 2x2x2 uses 7 multiplications.
    For 2x2x5, we can think of it as 5 independent 2x2x1 problems sharing A.
    
    Let's try: use 7 terms for cols 0-1 (Strassen), 7 terms for cols 2-3 (Strassen),
    but try to merge one term by having a single term contribute to both column pairs.
    """
    n, m, p = DIMENSIONS
    T = _build_target()
    R = 17
    
    # The tensor T[i,j,k] = 1 iff there exists i1,r,j2 with i=i1*m+r, j=r*p+j2, k=i1*n+j2
    # i.e., T[i,j,k] = 1 iff i1 = i//m, r = i%m, j2 = j%p, and k = i1*n + j2
    
    # For rank R, we need: T[i,j,k] = sum_{r=0}^{R-1} U[i,r]*V[j,r]*W[k,r]
    
    # Let's set up a system where we fix V and W to have a specific sparse structure
    # and solve for U, then verify.
    
    # Structure: 
    # - 7 terms for (cols 0,1) Strassen: indices 0-6
    # - 7 terms for (cols 2,3) Strassen: indices 7-13  
    # - 3 terms for col 4 (try to use only 3 instead of 4)
    
    # For col 4 schoolbook: C[i1,4] = A[i1,0]*B[0,4] + A[i1,1]*B[1,4]
    # This needs 2 terms minimum. But maybe we can share with Strassen terms.
    
    # Let's try: use the same U vectors from Strassen terms to also contribute to col 4.
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen block 1: cols 0,1
    # Term 0: (A00+A11)(B00+B11) -> contributes to C00, C11
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[1,1] = 1  # V[0*p+0], V[1*p+1]
    W[0,0] = 1; W[1,1] = 1  # W[0*n+0], W[1*n+1]
    
    # Term 1: (A10+A11)(B00-B10) -> contributes to C10, C11
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1  # V[0*p+0]
    W[1,1] = 1; W[1,1+1] = -1  # Wait, W indices: W[k] where k = i1*n+j2
    # W[1*n+0] = W[2] for C10, W[1*n+1] = W[3] for C11
    W[2,1] = 1; W[3,1] = -1
    
    # Term 2: A00*(B01-B11) -> contributes to C01
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1  # V[0*p+1], V[1*p+1]
    W[1,2] = 1; W[3,2] = 1  # W[0*n+1]=W[1], W[1*n+1]=W[3]
    
    # Term 3: A11*(B10-B00) -> contributes to C00, C10
    U[3,3] = 1
    V[5,3] = 1; V[0,3] = -1  # V[1*p+0], V[0*p+0]
    W[0,3] = 1; W[2,3] = 1  # W[0*n+0]=W[0], W[1*n+0]=W[2]
    
    # Term 4: (A00+A01)*B11 -> contributes to C00, C01
    U[0,4] = 1; U[1,4] = 1
    V[7,4] = 1  # V[1*p+1]
    W[0,4] = -1; W[1,4] = 1  # W[0*n+0], W[0*n+1]
    
    # Term 5: (A10-A00)*(B00+B01) -> contributes to C11
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1  # V[0*p+0], V[0*p+1]
    W[3,5] = 1  # W[1*n+1]
    
    # Term 6: (A01-A11)*(B10+B11) -> contributes to C00
    U[1,6] = 1; U[3,6] = -1
    V[5,6] = 1; V[7,6] = 1  # V[1*p+0], V[1*p+1]
    W[0,6] = 1  # W[0*n+0]
    
    # Strassen block 2: cols 2,3 (similar structure)
    base2 = 7
    # Term 7: (A00+A11)(B02+B13)
    U[0,7] = 1; U[3,7] = 1
    V[2,7] = 1; V[3,7] = 1  # V[0*p+2], V[1*p+3]
    W[2,7] = 1; W[3,7] = 1  # W[0*n+2], W[1*n+3]
    
    # Term 8: (A10+A11)(B02-B12)
    U[2,8] = 1; U[3,8] = 1
    V[2,8] = 1  # V[0*p+2]
    W[4,8] = 1; W[5,8] = -1  # W[1*n+2], W[1*n+3]
    
    # Term 9: A00*(B03-B13)
    U[0,9] = 1
    V[3,9] = 1; V[7,9] = -1  # V[0*p+3], V[1*p+3]
    W[3,9] = 1; W[5,9] = 1  # W[0*n+3], W[1*n+3]
    
    # Term 10: A11*(B12-B02)
    U[3,10] = 1
    V[7,10] = 1; V[2,10] = -1  # V[1*p+2], V[0*p+2]
    W[2,10] = 1; W[4,10] = 1  # W[0*n+2], W[1*n+2]
    
    # Term 11: (A00+A01)*B13
    U[0,11] = 1; U[1,11] = 1
    V[7,11] = 1  # V[1*p+3]
    W[2,11] = -1; W[3,11] = 1  # W[0*n+2], W[0*n+3]
    
    # Term 12: (A10-A00)*(B02+B03)
    U[2,12] = 1; U[0,12] = -1
    V[2,12] = 1; V[3,12] = 1  # V[0*p+2], V[0*p+3]
    W[5,12] = 1  # W[1*n+3]
    
    # Term 13: (A01-A11)*(B12+B13)
    U[1,13] = 1; U[3,13] = -1
    V[7,13] = 1; V[7,13] = 1  # V[1*p+2], V[1*p+3]
    W[2,13] = 1  # W[0*n+2]
    
    # Now for col 4: need C[i1,4] = A[i1,0]*B[0,4] + A[i1,1]*B[1,4]
    # We have 3 terms left (indices 14,15,16).
    # Try to use existing U vectors to also contribute to col 4.
    
    # Option: Add B[0,4] and B[1,4] to existing V vectors
    # Term 14: A00 * B04 -> C04
    U[0,14] = 1
    V[4,14] = 1  # V[0*p+4]
    W[4,14] = 1  # W[0*n+4]
    
    # Term 15: A10 * B04 -> C14  
    U[2,15] = 1
    V[4,15] = 1  # V[0*p+4]
    W[5,15] = 1  # W[1*n+4]
    
    # Term 16: (A01+A11) * B14 -> C04, C14
    U[1,16] = 1; U[3,16] = 1
    V[9,16] = 1  # V[1*p+4]
    W[4,16] = 1; W[5,16] = 1  # W[0*n+4], W[1*n+4]
    
    # Check this decomposition
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.max(np.abs(recon - T))
    
    if err < 1e-6:
        if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
            if _check_decomposition(U, V, W, T):
                return U, V, W
    
    # Try different combinations for col 4
    # Option 2: Use 3 terms with different structure
    # C04 = A00*B04 + A01*B14
    # C14 = A10*B04 + A11*B14
    
    # Term 14: A00*B04
    U[0,14] = 1; V[4,14] = 1; W[4,14] = 1
    # Term 15: A01*B14  
    U[1,15] = 1; V[9,15] = 1; W[4,15] = 1
    # Term 16: (A10+A11)*B04 -> but this gives C14 = A10*B04 + A11*B04, not what we want
    
    # Let me try a different approach: solve for the last 3 terms given the first 14 are fixed
    
    U_fixed = U[:, :14].copy()
    V_fixed = V[:, :14].copy()
    W_fixed = W[:, :14].copy()
    
    # The residual tensor after 14 terms:
    recon14 = np.einsum('ir,jr,kr->ijk', U_fixed, V_fixed, W_fixed)
    residual = T - recon14
    
    # Now we need: residual[i,j,k] = sum_{r=14}^{16} U[i,r]*V[j,r]*W[k,r]
    # This is a rank-3 approximation of the residual tensor.
    
    # Let's try to solve this by fixing V and W for the last 3 terms and solving for U
    # Or use a greedy approach.
    
    # For each output k, residual[:,:,k] should be a matrix of rank <= 3
    # (since we have 3 terms left)
    
    # Let's try specific structures for the last 3 terms:
    # We need to cover:
    # C04 = A00*B04 + A01*B14 (i=0, j=4 or j=9)
    # C14 = A10*B04 + A11*B14 (i=2 or i=3, j=4 or j=9)
    
    # The residual at k=4 (W[0*n+4]): should be A00*B04 + A01*B14
    # The residual at k=5 (W[1*n+4]): should be A10*B04 + A11*B14
    
    # Let's set up equations. For k=4:
    # residual[i,j,4] = sum_{r=14}^{16} U[i,r]*V[j,r]*W[4,r]
    # Non-zero residuals at k=4: (0,4,4), (0,9,4), (1,4,4), (1,9,4) maybe?
    
    # Actually, let me just compute what the residual should be and try to fit it.
    
    # For now, let's try a simpler approach: use ALS to find the last 3 terms
    # given the first 14 are fixed.
    
    U_try = np.zeros((n*m, R))
    V_try = np.zeros((m*p, R))
    W_try = np.zeros((p*n, R))
    U_try[:, :14] = U_fixed
    V_try[:, :14] = V_fixed
    W_try[:, :14] = W_fixed
    
    # Initialize last 3 terms randomly
    for r in range(14, 17):
        U_try[0, r] = 1.0
        V_try[4, r] = 1.0
        W_try[4, r] = 1.0
    
    # ALS for the last 3 terms only
    for it in range(20):
        # Update U for rows 0,1,2,3
        for i in range(n*m):
            # residual[i,:,:] = sum_{r=14}^{16} U[i,r]*V[j,r]*W[k,r]
            b = residual[i, :, :].flatten()
            A = np.zeros((m*p * p*n, 3))
            for r in range(3):
                outer = np.outer(V_try[:, 14+r], W_try[:, 14+r])
                A[:, r] = outer.flatten()
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_try[i, 14:17] = x
            except:
                pass
        
        # Update V for cols 4,9 (and maybe others)
        for j in range(m*p):
            b = residual[:, j, :].transpose(1, 0).flatten()
            A = np.zeros((n*m * p*n, 3))
            for r in range(3):
                outer = np.outer(U_try[:, 14+r], W_try[:, 14+r])
                A[:, r] = outer.flatten()
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_try[j, 14:17] = x
            except:
                pass
        
        # Update W for rows 4,5
        for k in range(p*n):
            b = residual[:, :, k].flatten()
            A = np.zeros((n*m * m*p, 3))
            for r in range(3):
                outer = np.outer(U_try[:, 14+r], V_try[:, 14+r])
                A[:, r] = outer.flatten()
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_try[k, 14:17] = x
            except:
                pass
        
        # Check convergence
        recon_try = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try)
        err_try = np.max(np.abs(recon_try - T))
        if err_try < 1e-6:
            # Round to half-integers
            U_r = np.round(2 * U_try) / 2
            V_r = np.round(2 * V_try) / 2
            W_r = np.round(2 * W_try) / 2
            if _check_decomposition(U_r, V_r, W_r, T):
                return U_r, V_r, W_r
            if _check_decomposition(U_try, V_try, W_try, T) and \
               _is_half_integer(U_try) and _is_half_integer(V_try) and _is_half_integer(W_try):
                return U_try, V_try, W_try
    
    return None

def _try_rank17_als(max_iters=50, attempts=30):
    """
    Try random initialization + ALS for rank 17, with half-integer rounding.
    """
    n, m, p = DIMENSIONS
    T = _build_target()
    R = 17
    start_time = time.time()
    deadline = start_time + 40  # 40 second budget
    
    for attempt in range(attempts):
        if time.time() > deadline:
            break
        
        # Random integer initialization
        U = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
        W = np.random.randint(-2, 3, size=(p*n, R)).astype(float)
        
        # Avoid zero columns
        for r in range(R):
            if np.all(U[:, r] == 0): U[0, r] = 1
            if np.all(V[:, r] == 0): V[0, r] = 1
            if np.all(W[:, r] == 0): W[0, r] = 1
        
        for it in range(max_iters):
            # ALS step for U
            U_new = np.zeros_like(U)
            for i in range(n*m):
                b = T[i].flatten()
                A = np.zeros((m*p * p*n, R))
                for r in range(R):
                    outer = np.outer(V[:, r], W[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i] = x
                except:
                    U_new[i] = U[i]
            U = U_new
            
            # ALS step for V
            V_new = np.zeros_like(V)
            for j in range(m*p):
                b = T[:, j, :].transpose(1, 0).flatten()
                A = np.zeros((n*m * p*n, R))
                for r in range(R):
                    outer = np.outer(U[:, r], W[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j] = x
                except:
                    V_new[j] = V[j]
            V = V_new
            
            # ALS step for W
            W_new = np.zeros_like(W)
            for k in range(p*n):
                b = T[:, :, k].flatten()
                A = np.zeros((n*m * m*p, R))
                for r in range(R):
                    outer = np.outer(U[:, r], V[:, r])
                    A[:, r] = outer.flatten()
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k] = x
                except:
                    W_new[k] = W[k]
            W = W_new
            
            # Check convergence
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-5:
                # Try to round to half-integers
                U_rounded = np.round(2 * U) / 2
                V_rounded = np.round(2 * V) / 2
                W_rounded = np.round(2 * W) / 2
                
                if _check_decomposition(U_rounded, V_rounded, W_rounded, T):
                    return U_rounded, V_rounded, W_rounded
                
                # Also check if already half-integer
                if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    if _check_decomposition(U, V, W, T):
                        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target()
    start_time = time.time()
    deadline = start_time + 110  # 110 second total budget (10s safety margin)
    
    # Try rank 17 with structured approach
    if time.time() < deadline - 50:
        result = _try_rank17_structured()
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 17 with random ALS
    if time.time() < deadline - 10:
        result = _try_rank17_als()
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Rank 18
    U, V, W = _build_rank18()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
