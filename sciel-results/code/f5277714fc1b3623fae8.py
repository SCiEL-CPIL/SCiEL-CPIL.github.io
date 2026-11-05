"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor T[i*m+j, j*p+k, k*n+i]=1."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p, atol=1e-8):
    """Verify exact decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T_true, atol=atol)

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) with integer entries.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U rows: [00, 01, 10, 11] (i*2+j)
    # V rows: [00, 01, 10, 11] (j*2+k)
    # W rows: [00, 01, 10, 11] (k*2+i)
    
    # r=0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # r=1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1; W[1, 1] = 1
    
    # r=2: M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # r=5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # r=6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_3x2x4_decomposition():
    """
    Schoolbook decomposition for 3x2x4 matrix multiplication.
    Rank = 3*2*4 = 24.
    Returns (U, V, W) with shapes (6, 24), (8, 24), (12, 24).
    Indexing: U[i*2+j, r], V[j*4+k, r], W[k*3+i, r]
    """
    n, m, p = 3, 2, 4
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _kron_decompose(n1, m1, p1, n2, m2, p2, U1, V1, W1, U2, V2, W2):
    """
    Combine two tensor decompositions via Kronecker product.
    If T1 has rank r1 and T2 has rank r2, the Kronecker product has rank r1*r2.
    
    For 3x4x8 = (3x2x4) ⊗ (1x2x2):
    n = n1*n2, m = m1*m2, p = p1*p2
    
    U_total shape: (n*m, r1*r2)
    V_total shape: (m*p, r1*r2)
    W_total shape: (p*n, r1*r2)
    """
    n, m, p = n1*n2, m1*m2, p1*p2
    r1, r2 = U1.shape[1], U2.shape[1]
    rank = r1 * r2
    
    U_total = np.zeros((n*m, rank))
    V_total = np.zeros((m*p, rank))
    W_total = np.zeros((p*n, rank))
    
    for r in range(rank):
        r1_idx = r // r2
        r2_idx = r % r2
        
        # U: index is (i1*m1+j1)*m2 + (i2*m2+j2) = i1*m1*m2 + i2*m2 + j1*m2 + j2
        # = (i1*n2 + i2)*m + j1*m2 + j2
        # U_total[(i1*m1+j1)*m2 + (i2*m2+j2), r] = U1[i1*m1+j1, r1_idx] * U2[i2*m2+j2, r2_idx]
        for i1 in range(n1):
            for j1 in range(m1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        idx = (i1*m1 + j1)*m2 + (i2*m2 + j2)
                        U_total[idx, r] = U1[i1*m1 + j1, r1_idx] * U2[i2*m2 + j2, r2_idx]
        
        # V: index is (j1*p1+k1)*p2 + (j2*p2+k2) = j1*p1*p2 + j2*p2 + k1*p2 + k2
        # V_total[(j1*p1+k1)*p2 + (j2*p2+k2), r] = V1[j1*p1+k1, r1_idx] * V2[j2*p2+k2, r2_idx]
        for j1 in range(m1):
            for k1 in range(p1):
                for j2 in range(m2):
                    for k2 in range(p2):
                        idx = (j1*p1 + k1)*p2 + (j2*p2 + k2)
                        V_total[idx, r] = V1[j1*p1 + k1, r1_idx] * V2[j2*p2 + k2, r2_idx]
        
        # W: index is (k1*n1+i1)*n2 + (k2*n2+i2) = k1*n1*n2 + k2*n2 + i1*n2 + i2
        # W_total[(k1*n1+i1)*n2 + (k2*n2+i2), r] = W1[k1*n1+i1, r1_idx] * W2[k2*n2+i2, r2_idx]
        for k1 in range(p1):
            for i1 in range(n1):
                for k2 in range(p2):
                    for i2 in range(n2):
                        idx = (k1*n1 + i1)*n2 + (k2*n2 + i2)
                        W_total[idx, r] = W1[k1*n1 + i1, r1_idx] * W2[k2*n2 + i2, r2_idx]
    
    return U_total, V_total, W_total

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return U, V, W

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Best fallback: schoolbook rank 96
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n * m * p
    
    # Strategy 1: Kronecker product (3x2x4) ⊗ (1x2x2 with Strassen)
    # 3x4x8 = (3x2) x (4x8) = (3x2x4) ⊗ (1x2x2)
    # Wait, that's not quite right. Let me think again.
    # 
    # Actually, the Kronecker product of matrix multiplications:
    # If C = A*B where A is (n1*n2) x (m1*m2) and B is (m1*m2) x (p1*p2),
    # we can write C as a block matrix and use the Kronecker structure.
    #
    # More precisely: if we have A = A1 ⊗ A2 (n1*n2 x m1*m2) and B = B1 ⊗ B2 (m1*m2 x p1*p2),
    # then C = A*B = (A1*B1) ⊗ (A2*B2).
    #
    # But our A is 3x4 and B is 4x8. We need 3 = n1*n2, 4 = m1*m2 = p1*m2... 
    # This doesn't factor cleanly.
    #
    # Alternative: 3x4x8. Note 4 = 2*2, 8 = 2*4.
    # We can write the multiplication as:
    # C[i, k] = sum_j A[i,j] B[j,k]
    # where i in [0,3), j in [0,4), k in [0,8).
    #
    # Let j = j1*2 + j2, k = k1*4 + k2, i = i1*... hmm.
    # 
    # Actually, let's use: 3x4x8. We can split the middle dimension:
    # j = 2*j1 + j2 where j1 in [0,2), j2 in [0,2)
    # k = 4*k1 + k2 where k1 in [0,2), k2 in [0,4)
    # i stays as is, i in [0,3)
    #
    # Then C[i, 4*k1+k2] = sum_{j1,j2} A[i, 2*j1+j2] B[2*j1+j2, 4*k1+k2]
    #
    # This is equivalent to: for each (i, k1, k2), C[i, 4*k1+k2] = sum_j A[i,j] B[j, 4*k1+k2]
    #
    # We can view this as: define A' as 3x4 (same as A) and B' as 4x8 (same as B).
    # The tensor is the same. The Kronecker approach requires the dimensions to factor.
    #
    # Let me try: 3x4x8 = (3x2x4) ⊗ (1x2x2)?
    # n = 3*1 = 3, m = 2*2 = 4, p = 4*2 = 8. Yes!
    # 
    # So we decompose:
    # - First factor: 3x2x4 (schoolbook rank 24)
    # - Second factor: 1x2x2 (this is just 2x2 multiplication, Strassen rank 7)
    # 
    # Wait, 1x2x2 means multiplying a 1x2 by a 2x2. That's just a row vector times a 2x2.
    # The tensor for 1x2x2: T[i*2+j, j*2+k, k*1+i] where i in [0,1), j in [0,2), k in [0,2).
    # T[j, 2*j+k, k] = 1. So T has 4 non-zero entries, rank 4 (schoolbook) or can we do better?
    # For 1x2x2, it's just computing [a,b] * [[c,d],[e,f]] = [ac+be, ad+bf].
    # This is rank 4 (each output element needs 2 products, 2 outputs). Can't do better than 4.
    #
    # Hmm, that gives rank 24*4 = 96, same as schoolbook. Not helpful.
    #
    # Let me try: 3x4x8 = (3x4x4) ⊗ (1x1x2)?
    # n = 3*1 = 3, m = 4*1 = 4, p = 4*2 = 8. Yes!
    # First factor: 3x4x4 (schoolbook rank 48)
    # Second factor: 1x1x2 (multiplying 1x1 by 1x2, rank 2)
    # Total rank: 48*2 = 96. Still same.
    #
    # The issue is that Kronecker product multiplies ranks. To get below 96,
    # we need at least one factor to have rank below its schoolbook.
    #
    # Let's try: 3x4x8. Can we write it as (3x2x2) ⊗ (1x2x4)?
    # n = 3*1 = 3, m = 2*2 = 4, p = 2*4 = 8. Yes!
    # First factor: 3x2x2 (schoolbook rank 12)
    # Second factor: 1x2x4 (schoolbook rank 8)
    # Total: 12*8 = 96. Same again.
    #
    # The fundamental issue: Kronecker product of schoolbook decompositions gives schoolbook rank.
    # We need to use a non-schoolbook decomposition for at least one factor.
    #
    # For 2x2x2, Strassen gives rank 7 < 8.
    # For 3x2x2, can we do better than 12? 
    # 3x2x2: multiply 3x2 by 2x2. C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # This is 3*2 = 6 independent "products" in schoolbook, but actually rank 12?
    # No wait. The tensor T[i*2+j, j*2+k, k*3+i] = 1.
    # i in [0,3), j in [0,2), k in [0,2). Total 12 entries, rank 12 schoolbook.
    # Can we do better? The lower bound is max(6, 4, 6) = 6. 
    # Actually for 3x2x2, the rank is known to be 12 (no improvement over schoolbook for these dims).
    #
    # Hmm, let me reconsider. The best known for small matrix multiplication:
    # 2x2x2: rank 7 (Strassen)
    # 3x3x3: rank 23 (Laser method)
    # For 3x2x2, I believe rank = 12 (no improvement).
    #
    # What about using Strassen on the 2x2 part?
    # 3x4x8: Let's split differently.
    # 
    # Actually, let me think about this more carefully.
    # The multiplication tensor for n×m×p has rank at most nmp (schoolbook).
    # Lower bound: max(nm, mp, pn).
    # For 3×4×8: lower bound = max(12, 32, 24) = 32.
    #
    # Known results:
    # - 2×2×2: rank 7
    # - 3×3×3: rank 23
    # - 4×4×4: rank 49 (or 48?)
    #
    # For 3×4×8, we can potentially use:
    # 3×4×8 = 3×(2×2)×(2×4) 
    # This suggests a Kronecker-like structure but not exact.
    #
    # Alternative approach: Use the fact that 8 = 2^3 and try to apply
    # Strassen-like methods.
    #
    # Actually, let me just try to use a Kronecker product where one factor
    # uses Strassen. 
    #
    # 3×4×8: Write as (3×2×4) ⊗ (1×2×2)?
    # n=3*1=3, m=2*2=4, p=4*2=8. 
    # But 1×2×2 has rank 4 (can't improve), so total = 24*4 = 96.
    #
    # What about (3×4×2) ⊗ (1×1×4)?
    # n=3, m=4, p=2*4=8. 
    # 3×4×2: schoolbook rank 24. 1×1×4: rank 4. Total 96.
    #
    # I think for 3×4×8, the rank might actually be 96 or close to it.
    # Let me just verify the schoolbook and try to find any improvement.
    #
    # Actually, wait. Let me reconsider the Kronecker product approach.
    # 
    # If we have two matrix multiplications:
    # C1 = A1 * B1 (n1×m1 × m1×p1)
    # C2 = A2 * B2 (n2×m2 × m2×p2)
    # 
    # Then the Kronecker product C1 ⊗ C2 = (A1 ⊗ A2) * (B1 ⊗ B2)
    # where A1⊗A2 is (n1*n2)×(m1*m2) and B1⊗B2 is (m1*m2)×(p1*p2).
    # 
    # So for 3×4×8, we need n1*n2=3, m1*m2=4, p1*p2=8.
    # Options:
    # (n1,n2) = (1,3) or (3,1)
    # (m1,m2) = (1,4), (2,2), (4,1)
    # (p1,p2) = (1,8), (2,4), (4,2), (8,1)
    # 
    # With constraint m1=p1 (since B1 is m1×p1 and must be square for Kronecker to work cleanly)?
    # No, B1 is m1×p1, not necessarily square.
    # 
    # Actually the constraint is just m1*m2=4 and p1*p2=8.
    # 
    # Let's try (n1,n2)=(3,1), (m1,m2)=(2,2), (p1,p2)=(4,2):
    # C1: 3×2 × 2×4 (rank 24 schoolbook)
    # C2: 1×2 × 2×2 (rank 4 schoolbook)
    # Total: 24*4 = 96.
    # 
    # Try (n1,n2)=(1,3), (m1,m2)=(4,1), (p1,p2)=(2,4):
    # C1: 1×4 × 4×2 (rank 8)
    # C2: 3×1 × 1×4 (rank 12)
    # Total: 8*12 = 96.
    # 
    # All give 96. The Kronecker product of schoolbook decompositions always gives schoolbook rank.
    #
    # To improve, we need a factor with rank < schoolbook.
    # The only small case with known improvement is 2×2×2 (rank 7 < 8).
    # 
    # Can we get a 2×2×2 factor? We need m1*m2=4, p1*p2=8, and one of the factors to be 2×2×2.
    # 2×2×2 requires n=2, m=2, p=2. But n1*n2=3, so neither n1 nor n2 can be 2.
    # 
    # So we can't directly use Strassen via Kronecker product for 3×4×8.
    #
    # Let me just go with schoolbook rank 96 and verify it works.
    # Then try gradient-based optimization to see if we can find rank < 96.
    
    # Verify schoolbook
    T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U_sb, V_sb, W_sb)
    assert np.allclose(T_recon, T_true, atol=1e-10), "Schoolbook verification failed"
    
    # Try to find lower rank using gradient descent
    # Start from rank 48 and try to reduce
    for r in [48, 44, 40, 36, 32]:
        if time.time() > deadline - 15:
            break
        
        # Initialize with schoolbook truncated + random
        rng = np.random.RandomState(42)
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        min_cols = min(r, n*m*p)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= r:
                        break
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        
        if r > min_cols:
            U[:, min_cols:] = rng.randn(n*m, r - min_cols) * 0.1
            V[:, min_cols:] = rng.randn(m*p, r - min_cols) * 0.1
            W[:, min_cols:] = rng.randn(p*n, r - min_cols) * 0.1
        
        # Gradient descent
        best_err = float('inf')
        for it in range(3000):
            if time.time() > deadline - 10:
                break
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            err = np.max(np.abs(R))
            
            if err < best_err:
                best_err = err
                if err < 1e-8:
                    U_r = _round_to_half_integer(U)
                    V_r = _round_to_half_integer(V)
                    W_r = _round_to_half_integer(W)
                    T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    err_r = np.max(np.abs(T_recon_r - T_true))
                    if err_r < 1e-8:
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = r
                        break
            
            if err < 1e-8:
                break
            
            eta = 0.02 / (1 + it * 0.001)
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
        
        if best_rank < r:
            break
    
    return best_result
# EVOLVE-BLOCK-END
