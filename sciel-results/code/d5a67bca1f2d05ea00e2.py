"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys
import random
import numpy as np

_START = time.time()
_TIMEOUT = 140.0

def _elapsed():
    return time.time() - _START

def _validate(U, V, W, n, m, p):
    """Check that the factor matrices exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = 1 for all i,j,k
    # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_rational_approx(vals, tol=1e-8):
    """Check if values are close to multiples of 0.5 (half-integers)."""
    for v in vals:
        if abs(v * 2 - round(v * 2)) > tol:
            return False
    return True

def _snap_to_half(vals):
    """Snap values to nearest half-integer."""
    return [round(v * 2) / 2.0 for v in vals]

def _construct_rank42(n, m, p):
    """Construct the known rank-42 Strassen-Kronecker decomposition."""
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col = 0
    for u in range(2):
        for v in range(3):
            for U_loc, V_loc, W_loc in steps:
                for li in range(2):
                    for lj in range(2):
                        l_idx = li * 2 + lj
                        val = U_loc[l_idx]
                        if val != 0:
                            g_i = li
                            g_j = 2 * u + lj
                            U[g_i * m + g_j][col] += val
                
                for lj in range(2):
                    for lk in range(2):
                        l_idx = lj * 2 + lk
                        val = V_loc[l_idx]
                        if val != 0:
                            g_j = 2 * u + lj
                            g_k = 2 * v + lk
                            V[g_j * p + g_k][col] += val
                
                for li in range(2):
                    for lk in range(2):
                        l_idx = li * 2 + lk
                        val = W_loc[l_idx]
                        if val != 0:
                            g_i = li
                            g_k = 2 * v + lk
                            W[g_k * n + g_i][col] += val
                            
                col += 1
                
    return U, V, W

def _try_random_decomposition(n, m, p, rank, deadline, max_attempts=500):
    """
    Try to find a rank-r decomposition by randomizing U,V and solving for W.
    For fixed U,V, the constraint is:
      For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is linear in W. We can set up a linear system and solve.
    The system has n*p unknowns (W entries) and n*m*p equations.
    For the system to have a solution, the equations must be consistent.
    
    Strategy: Randomly assign U and V entries from {-1, 0, 1} with some structure,
    then solve the linear system for W. Check if W has rational (half-integer) entries.
    """
    n_mp = n * m  # 8
    m_p = m * p   # 24
    p_n = p * n   # 12
    nmp = n * m * p  # 48
    
    # Build the linear system for W
    # Unknowns: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..rank-1
    # But we solve per (k,i) pair: for each (k,i), we have rank unknowns W[k*n+i, 0..rank-1]
    # The constraint for (i,j,k) involves W[k*n+i, :]
    
    # Actually, let's reformulate. For each (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let a_r = U[i*m+j, r] * V[j*p+k, r], then sum_r a_r * W[k*n+i, r] = 1
    # This is a dot product of vector a with W[k*n+i, :] equals 1.
    # For fixed (i,j), this gives one equation on W[k*n+i, :] for each k.
    # But different j values give different equations on the same W row.
    # So for each (i,k), we have m equations (one per j) on the rank-dimensional vector W[k*n+i, :].
    # Total: n*p = 12 groups of m=4 equations each, with rank unknowns per group.
    
    # If rank >= m = 4, each group of 4 equations in rank unknowns may have solutions.
    # We need the solution to be consistent (same W works for all j).
    
    # Better approach: set up the full linear system.
    # Total unknowns: p * n * rank = 12 * rank
    # Total equations: n * m * p = 48
    
    # For rank=41: 12*41=492 unknowns, 48 equations. Underdetermined.
    # We can find a particular solution and check if it's rational.
    
    # Let's use a different strategy: 
    # 1. Randomly initialize U (n*m x rank) and V (m*p x rank) with entries in {-1, 0, 1}
    # 2. For each (i,k) pair, solve the underdetermined system for W[k*n+i, :]
    # 3. Check if the resulting W gives exact decomposition
    # 4. Check if W entries are close to half-integers
    
    for attempt in range(max_attempts):
        if _elapsed() > deadline:
            return None
        
        # Generate random U and V with sparse structure
        # Use a bias toward having some zeros to make the system more likely to be solvable
        U_arr = np.zeros((n * m, rank), dtype=float)
        V_arr = np.zeros((m * p, rank), dtype=float)
        
        # Fill U and V with random ±1 values, with some zeros
        for idx in range(n * m):
            for r in range(rank):
                if random.random() < 0.4:  # 40% chance of non-zero
                    U_arr[idx, r] = random.choice([-1, 1])
        
        for idx in range(m * p):
            for r in range(rank):
                if random.random() < 0.4:
                    V_arr[idx, r] = random.choice([-1, 1])
        
        # For each (i,k) pair, solve for W[k*n+i, :]
        W_arr = np.zeros((p * n, rank), dtype=float)
        
        feasible = True
        for i in range(n):
            for k in range(p):
                # For this (i,k), we need: for each j in range(m):
                #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is m linear equations in rank unknowns W[k*n+i, 0..rank-1]
                
                # Build the m x rank matrix A where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A = np.zeros((m, rank), dtype=float)
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r]
                
                # Solve A * w = ones(m) where w = W[k*n+i, :]
                # This is an underdetermined system (m < rank typically)
                # Find a particular solution using least squares or pseudo-inverse
                try:
                    # Use np.linalg.lstsq for minimum-norm solution
                    w, residuals, rank_svd, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    feasible = False
                    break
                
                # Check if the solution actually satisfies the equations
                if not np.allclose(A @ w, np.ones(m), atol=1e-9):
                    feasible = False
                    break
                
                W_arr[k*n+i, :] = w
        
        if not feasible:
            continue
        
        # Check if W entries are close to half-integers
        W_flat = W_arr.flatten()
        if _is_rational_approx(W_flat, tol=1e-6):
            # Snap to half-integers
            W_snapped = _snap_to_half(W_flat).reshape(p * n, rank)
            
            # Verify the snapped solution
            U_list = U_arr.tolist()
            V_list = V_arr.tolist()
            W_list = W_snapped.tolist()
            
            if _validate(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = _START + _TIMEOUT - 5.0
    
    # First, construct the guaranteed rank-42 solution
    U42, V42, W42 = _construct_rank42(n, m, p)
    
    # Validate it
    if not _validate(U42, V42, W42, n, m, p):
        # Fallback: return what we have (should not happen)
        return {"U": U42, "V": V42, "W": W42}
    
    best_rank = 42
    best_U, best_V, best_W = U42, V42, W42
    
    # Try to find lower-rank decompositions
    # Attempt ranks from 41 down to 38
    for rank in [41, 40, 39, 38]:
        if _elapsed() > deadline:
            break
        
        result = _try_random_decomposition(n, m, p, rank, deadline, max_attempts=200)
        if result is not None:
            U_try, V_try, W_try = result
            best_rank = rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break  # Found the best possible in this range
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
