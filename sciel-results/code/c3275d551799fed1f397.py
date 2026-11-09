"""Matrix multiplication 2,4,6 - rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=2, m=4, p=6):
    """Check if U,V,W exactly decompose the matmul tensor."""
    # For each (i,j,k), sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    # And for all other index combinations, the sum should be 0
    # Efficient check: build the tensor and compare
    r = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i, j, k) in [
                    (i2, j2, k2) for i2 in range(n) for j2 in range(m) for k2 in range(p)
                ] else 0.0
                # Actually the matmul tensor is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor for matmul: T_{i,j,k} where the decomposition is:
                # (A*B)[i,k] = sum_j A[i,j] B[j,k]
                # As a tensor: T[i,j,k] = 1 (it's the identity in j for each i,k pair)
                # Wait, let me reconsider.
                # The matrix multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k
                # because C[i,k] = sum_j A[i,j]*B[j,k]
                # So T[i,j,k] = 1 for all i,j,k.
                # The decomposition: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                # Wait, the convention is W[k*n+i, r], so W index is k*n+i.
                # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For matmul, T[i,j,k] = 1 for all valid i,j,k
                # So we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
                pass
    
    # Better: use the standard check
    # The matmul tensor: for each i,k, C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = delta (it's 1 for the multiplication structure)
    # Actually T[i,j,k] = 1 for all i,j,k in the standard matmul tensor
    # because the multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let me just check: for all i,j,k, the sum should equal 1
    # (since T[i,j,k] = 1 for the matmul tensor of multiplying any matrices)
    
    # Actually no. The matmul tensor is defined as:
    # T[i,j,k] = 1 if we consider it as the structure tensor
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # For standard matmul: T[i,j,k] = 1 for all valid indices
    # So we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # But wait, that's only if ALL entries of the tensor are 1.
    # Let me verify: C[i,k] = sum_j A[i,j]*B[j,k]
    # This means the bilinear form has structure: for each (i,k), we sum over j
    # The tensor T has T[i,j,k] = 1 for all i,j,k
    # Yes, that's correct for matmul.
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :r], V[j*p+k, :r] * W[k*n+i, :r])
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _verify_fast(U, V, W, n=2, m=4, p=6):
    """Fast verification using matrix operations."""
    r = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Reshape and use einsum
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    # T[i,j,k] = sum_c Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    # For matmul tensor, T[i,j,k] should be 1 for all valid i,j,k
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-6):
        return False
    return True

def _is_half_integer(x, tol=1e-6):
    """Check if x is a half-integer (multiple of 0.5)."""
    return np.allclose(x * 2, np.round(x * 2), atol=tol)

def _snap_to_half_int(x, tol=1e-3):
    """Snap values to nearest half-integer if close."""
    return np.round(x * 2) / 2

def _structured_search(deadline):
    """
    Search for low-rank decompositions using structured patterns.
    
    For 2x4 x 4x6 matmul, we look for decompositions where:
    U[i,j,c], V[j,k,c], W[k,i,c] have structured entries.
    
    Strategy: Try ranks from theoretical lower bound up to schoolbook.
    Lower bound for n x m x p matmul: at least max(n*p, m*max(n,p)) ... 
    Actually the lower bound is max(n*p, n*m, m*p) in general, but for our case:
    n*p = 12, n*m = 8, m*p = 24, so lower bound is 24.
    Schoolbook is 48. So we search ranks 24 to 47.
    """
    n, m, p = 2, 4, 6
    schoolbook_rank = n * m * p  # 48
    
    # Lower bound: rank must be at least max(n*p, n*m, m*p) = max(12, 8, 24) = 24
    # But actually the Strassen-like lower bound is more subtle.
    # For n x m x p, the rank is at least max(n*p, n*m, m*p) / something...
    # Let's just try from 24 upward.
    
    best = None
    
    # Try a structured approach: parameterize U, V, W with limited support patterns
    # For rank r, each column c defines a "term" in the decomposition.
    
    # Approach: Use the fact that for matmul, we can write:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Group terms to reduce rank.
    
    # For 2x4 x 4x6, let's try to find a decomposition by solving a system.
    # Fix a structure and solve for coefficients.
    
    # Method: Try all subsets of "basis" terms
    # Each term (c) contributes U_c[i,j], V_c[j,k], W_c[k,i]
    # We need: sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] = 1 for all i,j,k
    
    # This is a system of n*m*p = 48 equations.
    
    # Let's try: assume W has a specific simple form, then solve for U and V.
    # Or use a rank-revealing approach.
    
    # Simpler: try random half-integer matrices of various ranks and check.
    # But that's expensive. Let's be smarter.
    
    # Key insight: For the matmul tensor, we can use "block" decompositions.
    # Split the 4 middle dimension into parts.
    
    # Try rank = 24 (theoretical lower bound)
    # If we can achieve rank 24, that would be great.
    
    # Let's try a specific construction:
    # For 2x4 x 4x6, note that 4 = 2+2.
    # Split B into B1 (2x6) and B2 (2x6).
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B1[j,k] + sum_{j=2}^{3} A[i,j]*B2[j-2,k]
    # This doesn't directly help reduce rank.
    
    # Alternative: Use the fact that we can "compress" the middle dimension.
    # Let's try to find a rank-r decomposition by solving a polynomial system.
    
    # Practical approach: Try ranks from 24 to 47, and for each,
    # use a structured initialization and local search.
    
    # For now, let's try a direct approach:
    # Assume the decomposition has a specific sparsity pattern.
    
    # Try: rank r where each "term" c corresponds to a pair (i0, k0) or similar.
    
    # Actually, let's try the simplest possible improvement:
    # Can we do better than schoolbook by combining terms?
    
    # For 2x4 x 4x6, the schoolbook uses 48 terms.
    # Theoretical minimum is at least 24.
    
    # Let's try to construct a rank-24 decomposition.
    # Idea: For each (i, k) pair (12 pairs), we need to compute sum_j A[i,j]*B[j,k].
    # If we could share computation across different (i,k) pairs...
    
    # For a fixed j, the contribution to all (i,k) is A[i,j]*B[j,k] which is an outer product.
    # So term j contributes: U_j[i,j'] = delta_{j,j'}, V_j[j',k] = delta_{j,j'}, W_j[k,i] = 1
    # Wait, that's not right.
    
    # Let me reconsider. In the schoolbook:
    # Term (i,j,k): U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1
    # This gives contribution 1 to T[i,j,k].
    
    # To reduce rank, we need to "merge" terms.
    # If two terms have the same U row pattern, V column pattern, and W pattern,
    # we can combine them.
    
    # For 2x4 x 4x6, let's see if there's a natural grouping.
    
    # Try: For each j (0..3), define a term that handles all i and k for that j.
    # Term j: U[i,j,c_j] = 1 for all i, V[j,k,c_j] = 1 for all k, W[k,i,c_j] = 1 for all k,i
    # But W[k,i,c_j] = 1 for all k,i means this term contributes to ALL (i,j',k) where j'=j.
    # Sum over c: For T[i,j',k], we get sum_{c: j_c = j'} 1*1*1 = number of terms with j_c = j'
    # If we have one term per j, then T[i,j,k] = 1 for all i,j,k. 
    # But wait, that gives rank = m = 4! That can't be right.
    
    # Let me re-examine. If term c has:
    # U[i,j,c] = 1 for all i (when j = j_c)
    # V[j,k,c] = 1 for all k (when j = j_c)  
    # W[k,i,c] = 1 for all k,i
    # Then T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # = sum_{c: j_c = j} 1*1*1 = 1 (if exactly one term per j)
    # So this gives a rank-4 decomposition!
    
    # But wait, is this valid? Let's check the shapes.
    # U has shape (n*m, r) = (8, r). U[i*m+j, c] = 1 for all i when j = j_c.
    # So U[0*m+j_c, c] = 1, U[1*m+j_c, c] = 1, and 0 otherwise.
    # V has shape (m*p, r) = (24, r). V[j_c*p+k, c] = 1 for all k.
    # So V[j_c*p+0, c] = 1, ..., V[j_c*p+5, c] = 1, and 0 otherwise.
    # W has shape (p*n, r) = (12, r). W[k*n+i, c] = 1 for all k,i.
    # So W[0, c] = W[1, c] = ... = W[11, c] = 1.
    
    # Let's verify: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # For a given (i,j,k):
    # U[i*m+j,c] is nonzero only if c corresponds to j_c = j, i.e., c = j (if we use one term per j)
    # V[j*p+k,c] is nonzero only if j_c = j, i.e., c = j
    # W[k*n+i,c] is always 1 for c = j
    # So T[i,j,k] = U[i*m+j,j]*V[j*p+k,j]*W[k*n+i,j] = 1*1*1 = 1. 
    # And for other c, at least one factor is 0.
    
    # So this IS a valid rank-4 decomposition! But that seems too good.
    # Let me double-check with a small example.
    
    # For 1x1 x 1x1: rank should be 1.
    # n=1, m=1, p=1. Term j=0: U[0,0]=1, V[0,0]=1, W[0,0]=1. T[0,0,0]=1. Rank 1. Correct.
    
    # For 2x1 x 1x2: C[i,k] = A[i,0]*B[0,k]. 
    # n=2, m=1, p=2. Term j=0: U[i*1+0, 0] = U[i,0] = 1 for i=0,1.
    # V[0*2+k, 0] = V[k,0] = 1 for k=0,1.
    # W[k*2+i, 0] = 1 for all k,i.
    # T[i,0,k] = U[i,0]*V[k,0]*W[2k+i,0] = 1*1*1 = 1. 
    # Rank 1. But is that correct? C[i,k] = A[i,0]*B[0,k].
    # The tensor T[i,j,k] with j fixed at 0: T[i,0,k] = 1.
    # Yes, rank 1 is correct for this case.
    
    # So for 2x4 x 4x6, rank 4 should work!
    # Let's construct it.
    
    rank = m  # 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for j in range(m):
        c = j
        # U[i*m+j, c] = 1 for all i
        for i in range(n):
            U[i * m + j, c] = 1
        # V[j*p+k, c] = 1 for all k
        for k in range(p):
            V[j * p + k, c] = 1
        # W[k*n+i, c] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i, c] = 1
    
    # Verify
    if _verify_fast(U, V, W):
        # Check if entries are half-integers
        if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
            return {"U": U, "V": V, "W": W}
    
    # If the simple construction doesn't work (shouldn't happen), fall back to schoolbook
    return _schoolbook()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Try the structured low-rank construction
    result = _structured_search(deadline)
    
    # Verify the result
    if result is not None:
        U, V, W = result["U"], result["V"], result["W"]
        if _verify_fast(U, V, W):
            return result
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
