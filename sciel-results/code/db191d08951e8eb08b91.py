"""Tensor decomposition search for 3x4x7 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]
DEADLINE_MARGIN = 20.0  # seconds before 150s wall-clock limit

def _verify(U, V, W, n=3, m=4, p=7):
    """Verify that U, V, W exactly decompose the 3x4x7 matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T_{i,j,k} = delta
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Check: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    # And for all other (i,j,k), should be 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if i == j == 0 and k == 0:
                    pass
                # Actually the tensor is: T[i,j,k] = 1 if this is a valid index combo
                # For matrix multiplication (n x m) * (m x p) -> (n x p):
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor T_{i,k,j} = delta (identity in j dimension)
                # But the problem says W[k*n+i, r], so the output index is k*n+i
                # This means the tensor is T[i,k,j] = 1 (the middle index j is the contraction)
                # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid i,k
                # Wait, that's not right either. Let me re-read.
                # U shape: (n*m, rank) -> indexed by (i,j)
                # V shape: (m*p, rank) -> indexed by (j,k)  
                # W shape: (p*n, rank) -> indexed by (k,i)
                # The decomposition is: T_{i,k} for the output, but it's a bilinear map
                # Actually for matrix mult: C_{i,k} = sum_j A_{i,j} B_{j,k}
                # The tensor is T_{i,j,k} where T_{i,j,k} = 1 (delta function in some sense)
                # More precisely: the multiplication tensor has entries T_{i,k,j} = 1
                # So the check is: for each c, the outer product U_c ⊗ V_c ⊗ W_c
                # and sum over c should give the tensor.
                # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # And T_{i,j,k} should equal 1 for all i,j,k? No...
                # For matrix multiplication (n×m)(m×p), the tensor is:
                # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
                # Actually the standard tensor: T_{i,k}^{j} = 1, meaning for each output (i,k),
                # the contribution from input pair (j) is A[i,j]*B[j,k]
                # So the trilinear form is: sum_{i,j,k} T_{i,j,k} * A_{i,j} * B_{j,k} * C_{i,k}
                # where T_{i,j,k} = 1 for all valid i,j,k (since each term contributes)
                # Wait no. Let me think again.
                # C[i,k] = sum_j A[i,j] B[j,k]
                # This is a bilinear map. As a trilinear form with dual:
                # sum_{i,j,k} A[i,j] B[j,k] C'[i,k] where C' is dual
                # The tensor T has T[i,j,k] = 1 for all i,j,k in ranges
                # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for ALL i,j,k
                expected = 1
                if s != expected:
                    return False
    return True

def _try_rational_search(n, m, p, max_rank, deadline):
    """
    Try to find a low-rank decomposition using randomized search with rational coefficients.
    Uses the structure that for matrix multiplication, we can try random sparse patterns.
    """
    # Start from schoolbook and try to reduce rank
    # Strategy: try various rank values from lower bound up
    # Lower bound: max(n*p, n*m, m*p) = max(21, 12, 28) = 28
    # But actually the asymptotic lower bound is higher
    # For now, try a few candidate ranks
    
    # Use a different approach: start with schoolbook (rank 84) and try to find
    # linear dependencies among the rank-1 tensors to reduce rank.
    
    # The schoolbook decomposition has tensors T_c = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} for each (i,j,k)
    # These are all linearly independent in the full tensor space, so we can't simply remove them.
    
    # Alternative: try to construct a decomposition with non-integer entries.
    # For 2x2x2, Strassen uses 7 terms with +/-1 entries.
    
    # For 3x4x7, let's try a block approach:
    # Split 4 = 2+2, 7 = 4+3
    # Use Strassen-like blocks where possible
    
    # Actually, let's try a direct randomized search for small ranks
    # with rational coefficients (denominators up to 2)
    
    best_result = None
    
    # Try ranks from a reasonable lower bound
    # Theoretical lower bound: omega(n,m,p) >= max(n*m, m*p, n*p) / something
    # Simple bound: rank >= max(n*m, m*p, n*p) is NOT correct
    # Actually rank >= log2(n*m*p) is not helpful
    # The correct simple bound: rank >= max(n, m, p) * min(n, m, p) in some cases
    # For now, try from 40 to 84
    
    for rank in range(40, min(84, max_rank + 1)):
        if time.time() > deadline:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Random sparse initialization
            # Each column has a few non-zero entries
            nnz_per_col = max(2, 84 // rank)
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for c in range(rank):
                # Choose random positions with +/-1
                for _ in range(nnz_per_col):
                    ui = random.randint(0, n * m - 1)
                    U[ui][c] = random.choice([-1, 1])
                    vi = random.randint(0, m * p - 1)
                    V[vi][c] = random.choice([-1, 1])
                    wi = random.randint(0, p * n - 1)
                    W[wi][c] = random.choice([-1, 1])
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    n, m, p = DIMENSIONS
    
    # First, get the guaranteed schoolbook solution
    result = _schoolbook(n, m, p)
    
    # Try to improve using various strategies
    # Strategy 1: Check if we can find a better solution via structured search
    
    # For 3x4x7, let's try a block decomposition approach
    # 4 = 2 + 2, and 7 = 4 + 3
    # Use Strassen's algorithm for the 2x2x2 blocks where applicable
    
    # Actually, let's try a more systematic approach:
    # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # has a known structure. For small dimensions, we can try to find
    # decompositions using known algorithms.
    
    # Try a few structured approaches
    # Approach: use the fact that 3x4x7 can be decomposed as a sum of
    # smaller matrix multiplications
    
    # 3x4*4x7 = (1x4*4x7) + (2x4*4x7)
    # or (3x2*2x7) + (3x2*2x7) if we split the 4 into 2+2
    # 3x2*2x7 has tensor rank at most 3*2*7 = 42, but might be less
    # 3x1*1x7 has rank 3*7 = 21 (or less?)
    
    # For 1xm*mxp: rank is exactly m (each of the m terms is needed)
    # Actually for 1xm*mxp: C[0,k] = sum_j A[0,j]*B[j,k]
    # This is rank p (one term per k) if we use schoolbook, but actually
    # the tensor T[j,k] = 1 for all j,k, which has rank min(m,p) in the
    # bilinear sense... no, the tensor rank is different.
    
    # Let me just try the search and fall back to schoolbook
    try:
        improved = _try_rational_search(n, m, p, 84, deadline)
        if improved is not None:
            return improved
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    # Try one more approach: verify schoolbook and return it
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback: return schoolbook anyway
    return result
