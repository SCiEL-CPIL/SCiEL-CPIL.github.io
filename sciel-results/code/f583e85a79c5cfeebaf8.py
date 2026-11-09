"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _verify_decomposition_np(U, V, W, n, m, p, rank):
    """Fast verification using numpy (floating point, for screening only)."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _try_rank_random(n, m, p, rank, n_trials=200, timeout=12.0):
    """Try to find a rank-r decomposition via random initialization + local search."""
    start = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    for trial in range(n_trials):
        if time.time() - start > timeout:
            break
        
        # Random initialization with small integers
        U = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(nm)]
        V = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(mp)]
        W = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(pn)]
        
        # Local search: iteratively adjust entries to reduce error
        best_err = float('inf')
        for it in range(50):
            if time.time() - start > timeout:
                break
            # Compute current error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        err += (val - 1)**2
            if err < 1e-10:
                # Check exact
                if _verify_decomposition(U, V, W, n, m, p, rank):
                    return U, V, W
            if err < best_err:
                best_err = err
            # Random perturbation
            mat = random.choice([U, V, W])
            row = random.randint(0, len(mat)-1)
            col = random.randint(0, rank-1)
            mat[row][col] += random.choice([-2,-1,1,2])
    
    return None

def _try_rank_gaussian(n, m, p, rank, timeout=15.0):
    """Try to find a rank-r decomposition using structured approach.
    Fix W to be identity-like and solve for U,V."""
    start = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy: set W[k*n+i, c] = delta_{k,0} for c < n, i.e., W has simple structure
    # Then the equation becomes: for k=0: sum_c U[i*m+j,c]*V[j*p,c]*W[i,c] = 1
    # This is getting complex. Let's try a different structured approach.
    
    # Approach: Use the fact that we can set some entries to 1 and solve linearly
    # For rank r, we need 105 trilinear equations with 71r variables.
    # Fix 105 - 71r + 1 variables arbitrarily (if feasible), solve the rest.
    # This only works if 71r >= 105, i.e., r >= 2 (always true for r>=35).
    # But the equations are trilinear, not linear, so this doesn't directly work.
    
    # Alternative: Fix U and V, then each equation in W is linear.
    # For fixed U,V: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is linear in W! So for fixed U,V, we have 105 linear equations in 21*r unknowns.
    # If 21*r >= 105, i.e., r >= 5, we can potentially solve.
    
    # So: randomly initialize U and V, then solve the linear system for W.
    # Then check if the solution is "nice" (integers or half-integers).
    
    for trial in range(50):
        if time.time() - start > timeout:
            break
        
        # Random U and V with small integer entries
        U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(nm)]
        V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(mp)]
        
        # Build linear system for W: 105 equations, 21*rank unknowns
        # Unknown: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,rank)
        # Equation (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Number of unknowns: pn * rank = 21 * rank
        # Number of equations: n*m*p = 105
        
        n_unknowns = pn * rank
        n_equations = n * m * p
        
        if n_unknowns < n_equations:
            continue  # Underdetermined won't work easily
        
        # Build A matrix (n_equations x n_unknowns) and b vector
        # A[e, u] where e is equation index, u is unknown index
        # Equation e corresponds to (i,j,k) = (e // (m*p), (e // p) % m, e % p)
        # Unknown u corresponds to (k', i', c) = (u // (n*rank), (u // rank) % n, u % rank)
        
        A = []
        b = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = [0] * n_unknowns
                    for c in range(rank):
                        # W[k*n+i, c] is at index (k*n+i)*rank + c
                        idx = (k * n + i) * rank + c
                        row[idx] = U[i*m+j][c] * V[j*p+k][c]
                    A.append(row)
                    b.append(1)
        
        # Solve using numpy least squares (since we may have more unknowns than equations)
        A_np = np.array(A, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        
        # Use lstsq for underdetermined system
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
        except:
            continue
        
        W = [[0] * rank for _ in range(pn)]
        for u in range(n_unknowns):
            k_idx = u // (n * rank)
            i_idx = (u // rank) % n
            c_idx = u % rank
            W[k_idx * n + i_idx][c_idx] = W_flat[u]
        
        # Check if W is close to integer/half-integer
        W_rounded = [[round(w * 2) / 2 for w in row] for row in W]
        
        # Verify with numpy
        if _verify_decomposition_np(U, V, W_rounded, n, m, p, rank):
            # Exact verification
            if _verify_decomposition(U, V, W_rounded, n, m, p, rank):
                return U, V, W_rounded
        
        # Also try rounding to integers
        W_int = [[round(w) for w in row] for row in W]
        if _verify_decomposition_np(U, V, W_int, n, m, p, rank):
            if _verify_decomposition(U, V, W_int, n, m, p, rank):
                return U, V, W_int
    
    return None

def _try_rank_structured(n, m, p, rank, timeout=15.0):
    """Try structured decompositions based on known patterns."""
    start = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy: For small matrices, try to find decompositions where
    # U, V, W have sparse structure.
    
    # One approach: partition the rank into blocks corresponding to different (i,k) pairs
    # For each (i,k) pair, we need sum_j,c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # If we set W[k*n+i,c] = 1 for specific c and 0 otherwise, we reduce to
    # sum_j U[i*m+j,c]*V[j*p+k,c] = 1 for those c.
    
    # Try: assign each (i,k) pair a dedicated set of rank columns
    # Number of (i,k) pairs: n*p = 21
    # For each (i,k), we need the m×m matrix product to give 1
    # If we use r_ik columns for (i,k), we need rank-m m×m decomposition
    # Minimum rank for 5×5 matrix mult is at least 5 (identity gives rank 5)
    
    # Total rank if we use 5 per (i,k): 21*5 = 105 (same as schoolbook)
    # To do better, we need to share columns between (i,k) pairs.
    
    # This is getting complex. Let's try a direct search with the linear approach
    # but with more trials and better initialization.
    
    for trial in range(100):
        if time.time() - start > timeout:
            break
        
        # Initialize U with mostly 1s and some structure
        U = [[random.choice([0, 1, 1, 1]) for _ in range(rank)] for _ in range(nm)]
        V = [[random.choice([0, 1, 1, 1]) for _ in range(rank)] for _ in range(mp)]
        
        # Solve for W
        n_unknowns = pn * rank
        n_equations = n * m * p
        
        if n_unknowns < n_equations:
            continue
        
        A = []
        b = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = [0] * n_unknowns
                    for c in range(rank):
                        idx = (k * n + i) * rank + c
                        row[idx] = U[i*m+j][c] * V[j*p+k][c]
                    A.append(row)
                    b.append(1)
        
        A_np = np.array(A, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
        except:
            continue
        
        W = [[0] * rank for _ in range(pn)]
        for u in range(n_unknowns):
            k_idx = u // (n * rank)
            i_idx = (u // rank) % n
            c_idx = u % rank
            W[k_idx * n + i_idx][c_idx] = W_flat[u]
        
        # Try rounding to half-integers
        W_rounded = [[round(w * 2) / 2 for w in row] for row in W]
        if _verify_decomposition(U, V, W_rounded, n, m, p, rank):
            return U, V, W_rounded
        
        # Try rounding to integers
        W_int = [[round(w) for w in row] for row in W]
        if _verify_decomposition(U, V, W_int, n, m, p, rank):
            return U, V, W_int
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Lower bound: r >= max(n*m, m*p, n*p) = max(15, 35, 21) = 35
    lower_bound = max(nm, mp, pn)
    
    # Start from lower bound and search upward
    for rank in range(lower_bound, nm * mp * pn + 1):
        if time.time() > deadline:
            break
        
        # Try different strategies for this rank
        result = _try_rank_gaussian(n, m, p, rank, timeout=min(10.0, deadline - time.time()))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline:
            break
        
        result = _try_rank_structured(n, m, p, rank, timeout=min(10.0, deadline - time.time()))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline:
            break
        
        result = _try_rank_random(n, m, p, rank, n_trials=50, timeout=min(10.0, deadline - time.time()))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank = nm * mp * pn
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
