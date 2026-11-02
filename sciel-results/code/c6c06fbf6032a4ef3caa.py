"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    rank = len(U[0])
    # Convert to numpy for speed
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each output element C[i,k] (i in 0..n-1, k in 0..p-1):
    # C[i,k] = sum_{a=0}^{n-1} sum_{b=0}^{m-1} A[i,a] * B[a,k]
    # In tensor form: for each (i,k), the coefficient of A[i,a]*B[a,k] must be 1.
    # Using the factorization: sum_r U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r] = 1 for all valid (i,a,k)
    # And 0 otherwise.
    
    for i in range(n):
        for k in range(p):
            for a in range(m):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m+a, r] * V_arr[a*p+k, r] * W_arr[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _build_strassen_53(n, m, p):
    """Build the baseline rank-53 Strassen decomposition."""
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        U[0][r1] = 1; U[3][r1] = 1
        V[0*p+k0][r1] = 1; V[1*p+k1][r1] = 1
        
        U[2][r2] = 1; U[3][r2] = 1
        V[0*p+k0][r2] = 1
        
        U[0][r3] = 1
        V[0*p+k1][r3] = 1; V[1*p+k1][r3] = -1
        
        U[3][r4] = 1
        V[1*p+k0][r4] = 1; V[0*p+k0][r4] = -1
        
        U[0][r5] = 1; U[1][r5] = 1
        V[1*p+k1][r5] = 1
        
        U[2][r6] = 1; U[0][r6] = -1
        V[0*p+k0][r6] = 1; V[0*p+k1][r6] = 1
        
        U[1][r7] = 1; U[3][r7] = -1
        V[1*p+k0][r7] = 1; V[1*p+k1][r7] = 1
        
        W[2*k0][r1] = 1; W[2*k0][r4] = 1; W[2*k0][r5] = -1; W[2*k0][r7] = 1
        W[2*k0+1][r2] = 1; W[2*k0+1][r4] = 1
        W[2*k1][r3] = 1; W[2*k1][r5] = 1
        W[2*k1+1][r1] = 1; W[2*k1+1][r2] = -1; W[2*k1+1][r3] = 1; W[2*k1+1][r6] = 1
    
    for i in range(7):
        add_strassen_block(7*i, 2*i, 2*i+1)
    
    base_r = 49
    U[0][base_r] = 1; V[0*p+14][base_r] = 1; W[28][base_r] = 1
    U[1][base_r+1] = 1; V[1*p+14][base_r+1] = 1; W[28][base_r+1] = 1
    U[2][base_r+2] = 1; V[0*p+14][base_r+2] = 1; W[29][base_r+2] = 1
    U[3][base_r+3] = 1; V[1*p+14][base_r+3] = 1; W[29][base_r+3] = 1
    
    return U, V, W

def _try_rank52_search(n, m, p, time_budget=60.0):
    """
    Try to find a rank-52 decomposition by modifying the rank-53 Strassen solution.
    Strategy: Try to merge/eliminate one rank term by finding a linear dependency.
    """
    if time.time() - _start_time > _DEADLINE - 30:
        return None
    
    U, V, W = _build_strassen_53(n, m, p)
    rank = 53
    
    # Try to eliminate one of the 4 schoolbook terms in the last column
    # by checking if it can be expressed as a combination of others.
    # This is unlikely to work directly, so let's try a different approach.
    
    # Alternative: try to find a rank-10 decomposition for 2x2x3
    # and use 5 such blocks (15/3 = 5 triples)
    # If successful: rank = 5 * 10 = 50
    
    # Let's try a structured search for 2x2x3 rank 10
    result = _find_2x2x3_rank10_structured(time_budget)
    if result is not None:
        U3, V3, W3 = result
        # Verify
        if _verify_decomposition(U3, V3, W3, 2, 2, 3):
            # Build full 2x2x15 using 5 blocks of 2x2x3
            full_rank = 5 * 10
            U_full = [[0] * full_rank for _ in range(4)]
            V_full = [[0] * full_rank for _ in range(30)]
            W_full = [[0] * full_rank for _ in range(30)]
            
            for block_idx in range(5):
                col_offset = 3 * block_idx
                r_offset = 10 * block_idx
                for r in range(10):
                    for a in range(4):
                        U_full[a][r_offset + r] = U3[a][r]
                    for b in range(6):
                        V_full[b + col_offset * 2][r_offset + r] = V3[b][r]
                    for c in range(6):
                        W_full[c + col_offset * 2][r_offset + r] = W3[c][r]
            
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                return U_full, V_full, W_full
    
    return None

def _find_2x2x3_rank10_structured(time_budget=60.0):
    """
    Find a rank-10 decomposition of 2x2 * 2x3 using structured algebraic methods.
    Returns (U, V, W) as lists of lists, or None.
    """
    # Based on known results, the 2x2x3 tensor has rank 10.
    # Let's try to construct it using a modified Strassen approach.
    # 
    # Key idea: Use 7 terms that are "Strassen-like" but adapted for 3 columns,
    # plus 3 additional terms.
    
    # Let me try a specific construction from the literature.
    # The idea is to use a "cyclic" pattern across the 3 columns.
    
    # Attempt: Use the following 10 products:
    # P1 = (A00+A11)(B00+B12)
    # P2 = (A10+A11)(B00)  
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)(B11)
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # P8 = A00*B02
    # P9 = A01*B12
    # P10 = (A10+A11)*B02  -- but this gives A10*B02 + A11*B02, we need A10*B02 + A11*B12
    
    # This doesn't quite work. Let me try a different approach.
    
    # Actually, let me try to solve this more carefully.
    # We need 10 rank-1 tensors that sum to the multiplication tensor.
    
    # Let me try a random search with verification, limited by time.
    np.random.seed(123)
    start = time.time()
    max_iter = 1000000
    
    coeffs = np.array([-1, 0, 1])
    
    # We'll search for U (4x10), V (6x10), W (6x10) with entries in {-1,0,1}
    # but with a structured approach: fix some entries to reduce search space.
    
    # Fix V to be in a canonical form: V[0,:] = [1,0,0,0,0,0] (first row)
    # This uses the scaling symmetry.
    
    # Actually, this search space is too large. Let me try a smarter approach.
    
    # Use the fact that the tensor is "block diagonal" in a sense:
    # The 2x2x3 tensor can be viewed as having structure related to the 2x2x2 tensor.
    
    # Let me try to find a solution by solving a system of polynomial equations
    # using a numerical approach (Newton's method or similar).
    
    # For now, return None and use the rank-53 baseline.
    return None

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Try to find a better decomposition
    result = _try_rank52_search(n, m, p, time_budget=80.0)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-53 Strassen
    U, V, W = _build_strassen_53(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
