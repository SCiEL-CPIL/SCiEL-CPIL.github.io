"""Matrix multiplication 3x4x6 rank minimization via structured search."""
import numpy as np
from itertools import product
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    where the output is C[i,k] = sum_j A[i,j]*B[j,k].
    
    The factorization is: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    We use a structured approach: try ranks from a lower bound upward, and for each rank
    try to find a valid decomposition using randomized initialization and local search.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor T[i,j,k] = 1 for all (i,j,k)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # Actually, the rank must be at least the matrix multiplication rank lower bound.
    # For n x m x p, known lower bounds include max(n*m, m*p, p*n) in general,
    # but for specific cases we can do better.
    # Trivial lower bound: rank >= max(n, m, p) since each factor has that many rows.
    # Better: rank >= ceil(n*m*p / (max singular values product)) but that's not helpful.
    # For 3x4x6: n*m=12, m*p=24, p*n=18. So rank >= 18? No, that's not right either.
    # The rank is at least the max of the three matrix multiplication dimensions in some sense.
    # Actually, rank >= max(n*m, m*p, p*n) is NOT a lower bound. 
    # The rank is at least the rank of any "slice" of the tensor.
    # T as a matrix of size (n*m) x (p) for fixed j: T[i,k; j] = 1, so rank 1.
    # So the tensor has many rank-1 slices.
    # The Strassen lower bound or similar might give rank >= something.
    # Let's just try from rank 1 upward, but we know schoolbook is 72.
    # Known best for 3x4x6 might be around 30-40.
    
    # Strategy: Use a parameterized family of decompositions and optimize.
    # For small ranks, use exhaustive or randomized search.
    
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    # Try to find the minimum rank
    # Start with a reasonable guess and work down
    # Schoolbook rank is 72
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Verify that U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, tol=1e-6):
        """Verify using numpy for speed."""
        rank = U.shape[1]
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape for batch computation
        # For each c, compute the outer product contribution
        # Actually, let's just do it directly with numpy broadcasting
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # We need to check that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # Build index arrays
        ii = np.repeat(np.arange(n), m*p)
        jj = np.tile(np.repeat(np.arange(m), p), n)
        kk = np.tile(np.arange(p), n*m)
        
        ui = U[ii*m+jj, :]  # (n*m*p, rank)
        vj = V[jj*p+kk, :]  # (n*m*p, rank)
        wk = W[kk*n+ii, :]  # (n*m*p, rank)
        
        # Sum over c
        result = np.sum(ui * vj * wk, axis=1)
        
        return np.allclose(result, np.ones(n*m*p), atol=tol)
    
    # Try ranks from a lower estimate upward
    # Known results: 3x4x6 matrix multiplication rank is believed to be around 30-40
    # Let's try from rank 20 upward
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # Generate schoolbook as fallback
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find better decompositions
    # Use a structured approach with integer/half-integer entries
    
    def try_rank(rank, max_attempts=200):
        """Try to find a valid decomposition of given rank."""
        nonlocal best_rank, best_decomp
        
        if rank >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return False
            
            # Randomized initialization with small integers and half-integers
            # Use values from {-1, -0.5, 0, 0.5, 1, 1.5, 2}
            vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
            
            U = np.random.choice(vals, size=(nm, rank))
            V = np.random.choice(vals, size=(mp, rank))
            W = np.random.choice(vals, size=(pn, rank))
            
            # Quick check
            if verify_decomposition_fast(U, V, W):
                best_rank = rank
                best_decomp = (U, V, W)
                return True
        
        return False
    
    # Also try a more structured approach: 
    # Use the fact that for matrix multiplication, there are known constructions.
    # For 3x4x6, we can try to build from known lower-rank constructions.
    
    # Try ranks from 20 to 71
    for rank in range(20, best_rank):
        if time.time() > deadline:
            break
        if try_rank(rank, max_attempts=50):
            # Found a valid decomposition at this rank, try to go lower
            break
    
    # If we found something better, convert to the required format
    if best_decomp is not None:
        U, V, W = best_decomp
        # Convert to list of lists with integer/half-integer entries
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        # Round to nearest half-integer if close
        for lst in [U_list, V_list, W_list]:
            for row in lst:
                for idx, val in enumerate(row):
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-6:
                        row[idx] = rounded
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
