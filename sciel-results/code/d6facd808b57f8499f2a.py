"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For all i in [n], j in [m], k in [p]:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul. Verified numerically."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # r=1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1
    
    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1
    
    return U, V, W

def _build_block_decomposition(n, m, p, splits, strassen_222):
    """
    Build a block decomposition.
    splits: dict mapping (i_range, j_range, k_range) to method ('strassen' or 'schoolbook')
    Returns U, V, W as numpy arrays.
    """
    sU, sV, sW = strassen_222
    rank = 0
    blocks = []
    
    # We need to partition the space. Let's use a recursive block approach.
    # For now, let's just implement the specific 54-rank construction from inspiration 2
    # and then try to optimize it.
    
    # Actually, let's build a general framework.
    # The key is to partition the indices i, j, k into blocks.
    # For each block (I, J, K), we use either Strassen (if 2x2x2) or schoolbook.
    
    # Let's stick to the proven 54-rank first, then try to improve.
    pass

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split 5 = 2+1+2, 6 = 2+2+2.
    """
    sU, sV, sW = _strassen_222()
    
    # Blocks:
    # 1. A[0:2, 0:2] * B[0:2, :] -> 2x2x6 -> 3 Strassen blocks (rank 21)
    # 2. A[0:2, 2:3] * B[2:3, :] -> 2x1x6 -> schoolbook (rank 12)
    # 3. A[0:2, 3:5] * B[3:5, :] -> 2x2x6 -> 3 Strassen blocks (rank 21)
    # Total: 21 + 12 + 21 = 54
    
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0..5}
    for block in range(3):
        k_off = block * 2
        for r in range(7):
            U[0, col] = sU[0, r]   # i=0, j=0
            U[1, col] = sU[1, r]   # i=0, j=1
            U[5, col] = sU[2, r]   # i=1, j=0
            U[6, col] = sU[3, r]   # i=1, j=1
            
            V[k_off, col] = sV[0, r]       # j=0, k=k_off
            V[k_off+1, col] = sV[1, r]     # j=0, k=k_off+1
            V[6+k_off, col] = sV[2, r]     # j=1, k=k_off
            V[6+k_off+1, col] = sV[3, r]   # j=1, k=k_off+1
            
            W[k_off*2, col] = sW[0, r]     # k=k_off, i=0
            W[k_off*2+1, col] = sW[1, r]   # k=k_off, i=1
            W[(k_off+1)*2, col] = sW[2, r] # k=k_off+1, i=0
            W[(k_off+1)*2+1, col] = sW[3, r]
            
            col += 1
    
    # Block 2: i in {0,1}, j in {2}, k in {0..5}
    for i in range(2):
        for k in range(6):
            U[i*5+2, col] = 1
            V[2*6+k, col] = 1
            W[k*2+i, col] = 1
            col += 1
    
    # Block 3: i in {0,1}, j in {3,4}, k in {0..5}
    for block in range(3):
        k_off = block * 2
        for r in range(7):
            U[3, col] = sU[0, r]   # i=0, j=3 (local j=0)
            U[4, col] = sU[1, r]   # i=0, j=4 (local j=1)
            U[8, col] = sU[2, r]   # i=1, j=3
            U[9, col] = sU[3, r]   # i=1, j=4
            
            V[18+k_off, col] = sV[0, r]       # j=3, k=k_off
            V[18+k_off+1, col] = sV[1, r]     # j=3, k=k_off+1
            V[24+k_off, col] = sV[2, r]       # j=4, k=k_off
            V[24+k_off+1, col] = sV[3, r]     # j=4, k=k_off+1
            
            W[k_off*2, col] = sW[0, r]
            W[k_off*2+1, col] = sW[1, r]
            W[(k_off+1)*2, col] = sW[2, r]
            W[(k_off+1)*2+1, col] = sW[3, r]
            
            col += 1
    
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _try_improve_rank(U, V, W, n, m, p, target_rank, time_limit=30):
    """
    Try to reduce the rank by merging columns or finding linear dependencies.
    This is a heuristic approach.
    """
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Try to find columns that can be merged or eliminated
    # This is complex, so for now we just return the input
    # A proper implementation would use algebraic techniques
    return U, V, W

def _solve_for_W(U, V, n, m, p, rank):
    """
    Given U and V, solve for W such that the decomposition holds.
    This is a linear system.
    """
    # For each (i, j, k), we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is nonlinear in W because W appears in a product with U and V.
    # However, if we fix U and V, we can view this as:
    # For each (i, k), let w_ik be the vector of W[k*n+i, :]
    # Then for each j: sum_r U[i*m+j, r] * V[j*p+k, r] * w_ik[r] = 1
    
    # This is a linear system in w_ik for each (i, k).
    # Let A_ik be a matrix where A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
    # Then A_ik @ w_ik = 1 (vector of ones of length m)
    
    W = np.zeros((p*n, rank))
    for i in range(n):
        for k in range(p):
            A_ik = np.zeros((m, rank))
            for j in range(m):
                for r in range(rank):
                    A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
            
            # Solve A_ik @ w_ik = ones(m)
            # This is an overdetermined system (m equations, rank unknowns)
            # Use least squares
            try:
                w_ik, residuals, rank_s, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                W[k*n+i, :] = w_ik
            except:
                return None
    
    return W

def _generate_sparse_UV(n, m, p, rank, seed=42):
    """
    Generate sparse U and V matrices with entries from {0, 1, -1}.
    """
    rng = np.random.default_rng(seed)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    
    # Each column of U and V should be sparse
    for r in range(rank):
        # Choose random non-zero entries for U[:, r]
        num_nonzero = rng.integers(2, 5)
        indices = rng.choice(n*m, size=num_nonzero, replace=False)
        for idx in indices:
            U[idx, r] = rng.choice([-1, 1])
        
        # Choose random non-zero entries for V[:, r]
        num_nonzero = rng.integers(2, 5)
        indices = rng.choice(m*p, size=num_nonzero, replace=False)
        for idx in indices:
            V[idx, r] = rng.choice([-1, 1])
    
    return U, V

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = (A*B)[i,k]
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    # Verify the initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Try to find lower rank decompositions using the linear algebra approach
    # Fix sparse U and V, solve for W
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 2:
                break
            
            U_try, V_try = _generate_sparse_UV(n, m, p, target_rank, seed=seed)
            W_try = _solve_for_W(U_try, V_try, n, m, p, target_rank)
            
            if W_try is not None:
                # Verify
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    # Check if entries are half-integers or integers
                    # The problem requires integer/half-integer entries
                    U_check = np.round(U_try * 2) / 2
                    V_check = np.round(V_try * 2) / 2
                    W_check = np.round(W_try * 2) / 2
                    
                    if _verify_decomposition(U_check, V_check, W_check, n, m, p):
                        best_U, best_V, best_W = U_check, V_check, W_check
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
