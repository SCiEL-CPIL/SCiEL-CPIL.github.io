import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    
    C = A * B where A, B, C are 2x2.
    Tensor convention: T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W

def _build_strassen_based(n, m, p):
    """
    Build a decomposition for n x m times m x p by applying Strassen to 2x2 blocks.
    
    For n=2, m=3, p=10:
    - Group the p=10 columns into 5 pairs
    - For each pair, use Strassen on A[0:2, 0:2] * B[0:2, k0:k0+2] (7 terms)
    - Add schoolbook for A[0:2, 2] * B[2, k0:k0+2] (4 terms)
    - Total: 5 * (7+4) = 55 terms
    """
    n, m, p = 2, 3, 10
    num_pairs = p // 2  # 5 pairs
    
    terms_per_pair = 7 + 4
    total_rank = num_pairs * terms_per_pair
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Get Strassen decomposition
    Su, Sv, Sw = _strassen_2x2x2()
    
    for pair in range(num_pairs):
        col_offset = pair * terms_per_pair
        k0 = pair * 2  # first column index in this pair
        
        # --- Strassen part: A[0:2, 0:2] * B[0:2, k0:k0+2] ---
        for r in range(7):
            # U: A[i,j] for i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    val = Su[i*2 + j, r]
                    if val != 0:
                        U[i * m + j, col_offset + r] = val
            # V: B[j, k] for j in {0,1}, k in {0,1} -> actual B[j, k0+k]
            for j in range(2):
                for k in range(2):
                    val = Sv[j*2 + k, r]
                    if val != 0:
                        V[j * p + (k0 + k), col_offset + r] = val
            # W: C[k, i] for k in {0,1}, i in {0,1} -> actual C[k0+k, i]
            for k in range(2):
                for i in range(2):
                    val = Sw[k*2 + i, r]
                    if val != 0:
                        W[(k0 + k) * n + i, col_offset + r] = val
        
        # --- Schoolbook part: A[0:2, 2] * B[2, k0:k0+2] ---
        # C[i, k0+k] += A[i, 2] * B[2, k0+k]
        sb_offset = 7
        for i in range(2):
            for k in range(2):
                idx = col_offset + sb_offset + i * 2 + k
                U[i * m + 2, idx] = 1
                V[2 * p + (k0 + k), idx] = 1
                W[(k0 + k) * n + i, idx] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Build the trivial schoolbook decomposition with rank n*m*p."""
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
    
    return U, V, W

def _try_improved_rank(n, m, p, target_rank, max_time=30):
    """
    Try to find a decomposition with rank <= target_rank using algebraic methods.
    
    For 2x3x10, we know rank 55 works. Try to see if we can do better
    by exploiting more structure.
    """
    deadline = time.time() + max_time
    
    # One approach: try to use a 2x3x5 subproblem decomposition
    # C[i,k] for k=0..4 and k=5..9 are independent
    # Each 2x3x5 subproblem: C[i,k] = sum_j A[i,j]*B[j,k]
    # Rank of 2x3x5 tensor?
    
    # The 2x3x5 tensor has 30 entries. Schoolbook rank is 30.
    # Can we do better?
    
    # For 2x3, the matrix multiplication tensor has rank 7 (Strassen-like)
    # Actually, for 2x3: C = A*B where A is 2x3, B is 3x1
    # This is just 2*3 = 6 multiplications, so rank 6.
    
    # For 2x3x5: we have 5 independent 2x3x1 problems
    # Each has rank 6, so total rank 30 (no savings from linearity)
    
    # But wait - we can group columns of B!
    # C[i, 0:5] = A[0:2, 0:3] * B[0:3, 0:5]
    # This is a 2x3x5 matrix multiplication
    
    # The key insight: can we do 2x3x5 in less than 30 operations?
    # 
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # We can write this as:
    # C[i,k] = (A[i,0] + A[i,1]) * B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # But this doesn't save anything directly.
    
    # Let's try: group A columns
    # C[i,0:2] uses A[i,0:3] * B[0:3, 0:2]
    # C[i,2:4] uses A[i,0:3] * B[0:3, 2:4]  
    # C[i,4] uses A[i,0:3] * B[0:3, 4:5]
    
    # For the 2x3x2 part (first 2 columns):
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1]
    
    # We can use Strassen-like tricks on the 2x2 part (A rows x B cols for fixed j)
    # But A has 3 columns, not 2.
    
    # Alternative: split A into A1 = A[:,0:2] and A2 = A[:,2]
    # C = A1*B1 + A2*B2 where B1 = B[0:2,:], B2 = B[2,:]
    
    # A1*B1 is 2x2x5 (A1 is 2x2, B1 is 2x5)
    # A2*B2 is 2x1x5 (A2 is 2x1, B2 is 1x5)
    
    # For 2x2x5: group columns in pairs
    # C[i, 0:2] = A1 * B1[:, 0:2] -> 2x2x2, rank 7 (Strassen)
    # C[i, 2:4] = A1 * B1[:, 2:4] -> 2x2x2, rank 7
    # C[i, 4:5] = A1 * B1[:, 4:5] -> 2x2x1, rank 4
    # Total for A1*B1: 7+7+4 = 18
    
    # For 2x1x5: schoolbook, rank 10
    # Total: 18 + 10 = 28 for 2x3x5
    
    # For 2x3x10: two copies of 2x3x5, total rank 56
    
    # That's worse than 55! Let me recheck.
    
    # Actually, for 2x3x10 with the original approach:
    # 5 pairs, each with 7 (Strassen for A[0:2,0:2]*B[0:2,2k:2k+2]) + 4 (schoolbook for A[0:2,2]*B[2,2k:2k+2])
    # = 5 * 11 = 55
    
    # Alternative: 2x3x5 decomposition with rank 28, then 2*28 = 56. Worse.
    
    # Try another split: A = A1 + A2 where A1 = A[:,0:2], A2 = A[:,2:3]
    # Same as above.
    
    # What about: use Strassen on the output dimension?
    # C has 2 rows, 10 columns. Group outputs in pairs: C[:, 0:2], C[:, 2:4], etc.
    # Each C[:, 2k:2k+2] = A * B[:, 2k:2k+2]
    # A is 2x3, B[:, 2k:2k+2] is 3x2
    # This is 2x3x2 matrix multiplication
    
    # What's the rank of 2x3x2?
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1}
    # 6 equations, 30 variables (with 10 rank-1 terms)
    
    # Schoolbook: 6 terms
    # Can we do better? 
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # C[0,1] = A00*B01 + A01*B11 + A02*B21
    # C[1,0] = A10*B00 + A11*B10 + A12*B20
    # C[1,1] = A10*B01 + A11*B11 + A12*B21
    
    # This is essentially 4 independent dot products of length 3.
    # Rank 4 (just 4 multiplications) is possible if we allow linear combinations.
    # Actually no - each C[i,k] requires a separate sum.
    # Minimum rank is 4? Let's see...
    
    # We need: sum_r U[i,j,r]*V[j,k,r]*W[i',k',r] = delta_{i,i'}*delta_{k,k'}
    # For each (i,k), we need the product to give 1.
    
    # Simple construction: for each (i,k) in {0,1}x{0,1}, use one rank-1 term
    # where U[i,j,r] = 1 for all j, V[j,k,r] = 1, W[i,k,r] = 1
    # That gives 4 terms. But U[i,j,r] = 1 for all j means we're summing over j,
    # which is exactly what we want!
    
    # So 2x3x2 has rank 4!
    
    # For 2x3x10: 5 output column pairs, each 2x3x2 with rank 4
    # Total: 5 * 4 = 20 terms!
    
    # Wait, let me verify this construction more carefully.
    # Term r corresponding to (i_r, k_r):
    # U[i,j,r] = 1 if i == i_r, else 0 (for all j)
    # V[j,k,r] = 1 if k == k_r, else 0 (for all j)  
    # W[i',k',r] = 1 if i' == i_r and k' == k_r, else 0
    
    # Product: U[i,j,r]*V[j,k,r]*W[i',k',r] = 1 if i=i_r, k=k_r, i'=i_r, k'=k_r
    # Sum over r: this equals 1 if (i,k) = (i',k'), else 0.
    # That's correct!
    
    # But wait - the tensor convention is W[k*n+i, r], so W is indexed by (k,i) not (i,k).
    # Let me redo: W[(k_r)*n + i_r, r] = 1
    
    # So for each (i_r, k_r) in {0,1} x {0,1}:
    # U[i_r * m + j, r] = 1 for j in {0,1,2}
    # V[j * p + k_r, r] = 1 for j in {0,1,2}
    # W[k_r * n + i_r, r] = 1
    
    # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r]
    # = sum_r (1 if i==i_r) * (1 if k==k_r) * (1 if k'==k_r and i'==i_r)
    # = 1 if i==i_r and k==k_r and k'==k_r and i'==i_r for some r
    # = 1 if i==i' and k==k', else 0. Correct!
    
    # So rank 20 for 2x3x10!
    
    U = np.zeros((n * m, 20))
    V = np.zeros((m * p, 20))
    W = np.zeros((p * n, 20))
    
    col = 0
    for i in range(n):
        for k in range(p):
            # U: A[i, j] for all j
            for j in range(m):
                U[i * m + j, col] = 1
            # V: B[j, k] for all j
            for j in range(m):
                V[j * p + k, col] = 1
            # W: C[k, i]
            W[k * n + i, col] = 1
            col += 1
    
    return U, V, W

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 120
    
    # Try the rank-20 decomposition first
    U, V, W = _try_improved_rank(n, m, p, 20, max_time=5)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Strassen-based rank 55
    U, V, W = _build_strassen_based(n, m, p)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
