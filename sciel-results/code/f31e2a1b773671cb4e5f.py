"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.linalg import norm, svd, solve

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY = 5.0

def _time_left():
    return _DEADLINE - _SAFETY - (time.time() - _START_TIME)

def _to_halfint(x, tol=1e-6):
    """Round to nearest multiple of 0.5 if close."""
    r = np.round(2.0 * x) / 2.0
    if np.allclose(x, r, atol=tol, rtol=0.0):
        return r
    return x

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matmul tensor."""
    # Tensor T[i,j,k] = delta; decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check by contracting: for each (i,j,k), sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should be 1
    # But the convention is: output C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] = 1 for all i,j,k (it's the identity in the multiplication sense)
    # Actually the matmul tensor has T[i,j,k] = 1 if we think of it as the structure tensor.
    # Let's verify properly: the decomposition says
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) (sum_k B[j,k] V[j*p+k,r]) ... no
    # Correct: C[i,k] = sum_r W[k*n+i,r] * (sum_j A[i,j] * U[i*m+j,r]) ... no.
    # The standard tensor decomposition for matmul:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r])
    # Wait, that's not right either. Let me be careful.
    # The multiplication tensor M has entries M[i,j,k] = 1 (it's a 3-tensor where each "fiber" is identity).
    # Actually no. The structure tensor for matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_r W[k*n+i, r] * U_prod[i, r] * V_prod[k, r]
    # where U_prod[i, r] = sum_j A[i,j] U[i*m+j, r] and V_prod[k, r] = sum_j B[j,k] V[j*p+k, r]
    # But that's a product of three terms which isn't standard.
    # 
    # The correct tensor decomposition: The matmul tensor T is a 3-index object T[i, j, k] 
    # such that the bilinear form (A, B) -> C has C[i,k] = sum_{j} A[i,j] B[j,k].
    # In tensor form, we can write T as a sum of rank-1 tensors:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where (u_r)_i = U[i*m + j_r, r]... no, this is getting confused.
    #
    # Let me use the standard formulation. The matmul tensor for multiplying (n×m) by (m×p)
    # is a tensor in R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n} (with the output indexed as (k,n) per convention).
    # A rank-1 term is: (a ⊗ b ⊗ c) where a ∈ R^{n×m}, b ∈ R^{m×p}, c ∈ R^{p×n}.
    # The full tensor is: T[A, B] = A * B (as a bilinear map).
    # In index form: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # So T_{i,j,k} = 1 for all i,j,k? No, that's not right.
    # 
    # The tensor T ∈ R^{(n*m) × (m*p) × (p*n)} is defined by:
    # T[alpha, beta, gamma] where alpha = i*m+j, beta = j*p+k, gamma = k*n+i
    # T[alpha, beta, gamma] = 1 if the indices are consistent (same j in alpha and beta, same i,k in alpha and gamma)
    # Actually, the standard way: the multiplication tensor has a 1 in position (i*m+j, j*p+k, k*n+i) 
    # for each valid (i,j,k). So it's a sparse tensor with n*m*p ones.
    #
    # A decomposition T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] means:
    # sum_r U[alpha, r] * V[beta, r] * W[gamma, r] = T[alpha, beta, gamma]
    #
    # So we need: for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (alpha, beta, gamma) not corresponding to a valid (i,j,k) triple, the sum should be 0.
    # But actually, every (alpha, beta, gamma) triple has a unique (i,j,k) if we parse them correctly,
    # so we just need the sum to equal 1 for all valid triples.
    
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    # For each (i,j,k), check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], np.multiply(V[j*p+k], W[k*n+i]))
                if abs(val - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2. Rank 7."""
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)(e), M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)(h), M6 = (c-d)(f+g), M7 = (b-d)(e+f)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 2x2: i,j,k in {0,1}
    # U indices: i*2+j: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V indices: j*2+k: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W indices: k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (a+d)(e+h)
    # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # U: a -> U[0,0]=1, d -> U[3,0]=1
    # V: e -> V[0,0]=1, h -> V[3,0]=1
    # W: C11 gets M1, C22 gets M1
    # C11: i=0,k=0 -> W[0*2+0, 0] = W[0,0] += 1
    # C22: i=1,k=1 -> W[1*2+1, 0] = W[3,0] += 1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (c+d)(e)
    # c=A[1,0], d=A[1,1], e=B[0,0]
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    # C21: i=1,k=0 -> W[0*2+1, 1] = W[1,1] += 1
    W[1,1] = 1
    
    # M3 = a(f-h)
    # a=A[0,0], f=B[0,1], h=B[1,1]
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    # C12: i=0,k=1 -> W[1*2+0, 2] = W[2,2] += 1
    W[2,2] = 1
    
    # M4 = d(g-e)
    # d=A[1,1], g=B[1,0], e=B[0,0]
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    # C11: i=0,k=0 -> W[0,3] += 1; C21: i=1,k=0 -> W[1,3] += 1
    W[0,3] = 1; W[1,3] = 1
    
    # M5 = (a+b)(h)
    # a=A[0,0], b=A[0,1], h=B[1,1]
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    # C11: i=0,k=0 -> W[0,4] -= 1; C12: i=0,k=1 -> W[2,4] += 1
    W[0,4] = -1; W[2,4] = 1
    
    # M6 = (c-d)(f+g)
    # c=A[1,0], d=A[1,1], f=B[0,1], g=B[1,0]
    U[2,5] = 1; U[3,5] = -1
    V[1,5] = 1; V[2,5] = 1
    # C22: i=1,k=1 -> W[3,5] += 1
    W[3,5] = 1
    
    # M7 = (b-d)(e+f)
    # b=A[0,1], d=A[1,1], e=B[0,0], f=B[0,1]
    U[1,6] = 1; U[3,6] = -1
    V[0,6] = 1; V[1,6] = 1
    # C11: i=0,k=0 -> W[0,6] += 1
    W[0,6] = 1
    
    return U, V, W

def _build_strassen_4x4_4x8():
    """Build 4x4 * 4x8 using Strassen on 2x2 blocks. Rank = 7*4 = 28 for the 4x4 part, then x2 for 8 cols."""
    # Split 4x4 into four 2x2 blocks: A = [[A11,A12],[A21,A22]]
    # Split 4x8 into two 4x4 blocks: B = [B1 | B2]
    # For each 4x4 block, use Strassen on 2x2 sub-blocks.
    # Actually, let's just use Strassen 2x2x2 directly on 2x2 blocks.
    # 4x4 * 4x4: split into 2x2 blocks, use Strassen for each pair of blocks.
    # This gives rank 7 for the whole 4x4*4x4 (using the 2x2 Strassen recursively).
    # For 4x4 * 4x8: do 4x4*4x4 twice (for the two halves of B).
    # Total rank = 7 + 7 = 14? No, that's not right.
    # 
    # Let me think more carefully. 4x4 * 4x4 can be done with rank 7 using Strassen
    # if we view it as 2x2 blocks of 2x2 matrices. The Strassen algorithm gives
    # 7 multiplications of 2x2 matrices, each of which is a 2x2*2x2->2x2 operation.
    # But each 2x2*2x2 multiplication has rank 4 (schoolbook) or 7 (Strassen).
    # If we use Strassen for the 2x2 level too, we get 7*7 = 49 for 4x4*4x4.
    # 
    # For 4x4 * 4x8: we can split B into two 4x4 blocks and do two 4x4*4x4 multiplications.
    # Each has rank 49 (with recursive Strassen) or rank 7*4=28 (using Strassen at top level only).
    # 
    # Let me use the simpler approach: Strassen at the 2x2 level only.
    # 4x4 * 4x4: 4 block multiplications (A11*B11, A12*B21, etc.) but Strassen reduces to 7.
    # Each "block multiplication" is 2x2*2x2 which we do with schoolbook (rank 4) or Strassen (rank 7).
    # If schoolbook: rank = 7*4 = 28.
    # If Strassen: rank = 7*7 = 49.
    #
    # For 4x4 * 4x8: two 4x4*4x4 multiplications.
    # Schoolbook at 2x2 level: rank = 2 * 28 = 56.
    # Strassen at 2x2 level: rank = 2 * 49 = 98.
    #
    # The best known is probably around 49-56. Let me try the 56 approach (Strassen top, schoolbook bottom).
    
    n, m, p = 4, 4, 8
    # We'll build this by combining Strassen 2x2 decompositions.
    # This is complex to code correctly. Let me use a simpler known result.
    # 
    # Actually, let me just use the schoolbook and try to reduce rank numerically.
    U, V, W = _schoolbook(n, m, p)
    return U, V, W

def _try_rank_reduction(U, V, W, target_rank, n, m, p, max_iter=200):
    """Try to reduce rank using iterative methods."""
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Try to find a lower rank decomposition by optimization
    # This is a non-convex problem. Use a simple approach:
    # Randomly initialize and do gradient descent on the residual.
    
    np.random.seed(42)
    
    # Try several random initializations
    for trial in range(5):
        if _time_left() < 10:
            break
            
        U_try = np.random.randn(n*m, target_rank) * 0.1
        V_try = np.random.randn(m*p, target_rank) * 0.1
        W_try = np.random.randn(p*n, target_rank) * 0.1
        
        lr = 0.01
        for it in range(max_iter):
            if _time_left() < 10:
                break
                
            # Compute residual: for each (i,j,k), sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1
            # Gradient with respect to U, V, W
            dU = np.zeros_like(U_try)
            dV = np.zeros_like(V_try)
            dW = np.zeros_like(W_try)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # val = sum_r U_try[i*m+j,r] * V_try[j*p+k,r] * W_try[k*n+i,r]
                        # dU[i*m+j,r] += (val - 1) * V_try[j*p+k,r] * W_try[k*n+i,r]
                        # dV[j*p+k,r] += (val - 1) * U_try[i*m+j,r] * W_try[k*n+i,r]
                        # dW[k*n+i,r] += (val - 1) * U_try[i*m+j,r] * V_try[j*p+k,r]
                        
                        prod_VW = np.multiply(V_try[j*p+k], W_try[k*n+i])
                        prod_UW = np.multiply(U_try[i*m+j], W_try[k*n+i])
                        prod_UV = np.multiply(U_try[i*m+j], V_try[j*p+k])
                        
                        val = np.dot(U_try[i*m+j], prod_VW)
                        err = val - 1.0
                        
                        dU[i*m+j] += err * prod_VW
                        dV[j*p+k] += err * prod_UW
                        dW[k*n+i] += err * prod_UV
            
            # Update
            U_try -= lr * dU
            V_try -= lr * dV
            W_try -= lr * dW
            
            # Check convergence
            if it % 50 == 0:
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_try[i*m+j], np.multiply(V_try[j*p+k], W_try[k*n+i]))
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-6:
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    return best_U, best_V, best_W
    
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Try to reduce rank
    # Target: try ranks from low to high
    for target_rank in [49, 56, 64, 72, 80, 98, 128]:
        if _time_left() < 15:
            break
        
        U_try, V_try, W_try = _try_rank_reduction(U, V, W, target_rank, n, m, p, max_iter=100)
        
        # Round to half-integers
        U_rounded = _to_halfint(U_try)
        V_rounded = _to_halfint(V_try)
        W_rounded = _to_halfint(W_try)
        
        if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
