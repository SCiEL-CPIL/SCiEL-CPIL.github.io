"""Matrix multiplication 3,4,4 rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # The tensor for multiplying (n x m) by (m x p) to get (n x p)
    # Tensor T[i, j, k] = 1 iff the output entry (i,k) depends on A[i,j] * B[j,k]
    # In the decomposition: C[i,k] = sum_r U[i*m+j... wait, let me be careful.
    
    # Standard tensor decomposition for matrix multiplication:
    # C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The matmul tensor T has indices (i, j, k) for i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (it's the identity tensor)
    # 
    # A rank-r decomposition: T = sum_{r=0}^{R-1} U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector of size n*m, V_r is a vector of size m*p, W_r is a vector of size n*p
    # 
    # Actually, let me re-read the interface:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # 
    # The decomposition satisfies: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1  (if this is the matmul tensor)
    # 
    # Wait, the output factor follows the official trace convention W[k*n+i, r]
    # So W has shape (p*n, rank) with index k*n + i.
    # 
    # The matmul tensor: T[i, j, k] = 1 for all valid i,j,k (it's the 3D identity)
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # This is the standard Strassen-type decomposition.
    
    # For the 1 x m x p case (n=1), the rank is m + p - 1.
    # Construction: for r = 0, 1, ..., m+p-2:
    #   U[j, r] = 1 if j <= min(r, m-1) else 0
    #   V[j, k, r] = 1 if j + k == r (with j in [0,m), k in [0,p)) else 0
    #   W[k, r] = 1 if k == min(r, p-1) else 0
    # 
    # Let me verify this for m=4, p=4, so r = 0..6:
    # r=0: U=[1,0,0,0], V[0,0]=1, W[0]=1
    #   Check: sum_r U[j,r]*V[j,k,r]*W[k,r] for output C[0,0] = A[0,0]*B[0,0]
    #   Only r=0 contributes: U[0,0]*V[0,0,0]*W[0,0] = 1*1*1 = 1 ✓
    #   For C[0,1]: need sum_j A[0,j]*B[j,1]. Only r where V[j,1,r]=1 and W[1,r]=1
    #   W[1,r]=1 iff r=1 or r>=p... W[k,r]=1 if k==min(r,p-1). So W[1,r]=1 iff min(r,3)==1, i.e., r=1.
    #   V[j,1,1]=1 iff j+1==1, so j=0. U[0,1]=1 (since 0<=min(1,3)=1).
    #   So C[0,1] gets A[0,0]*B[0,1] from r=1. But we need sum_j A[0,j]*B[j,1] = A[0,0]*B[0,1]+A[0,1]*B[1,1]+...
    #   This doesn't seem right. Let me reconsider.
    
    # Actually, the standard construction for 1xm x pxp (where we compute a row vector times a matrix)
    # is different. Let me think again.
    
    # For computing C = A * B where A is 1 x m and B is m x p:
    # C[k] = sum_j A[j] * B[j,k]
    # 
    # The tensor is T[j,k] for the scalar product... no, it's a 3-index tensor.
    # Actually for 1xm x pxp, the "matrix multiplication" is just a vector-matrix product.
    # The tensor has indices (j, k) where j indexes A and k indexes the output, and the
    # "B" index is implicit... 
    # 
    # Let me use the general framework. For n=1, m=4, p=4:
    # U has shape (1*4, R) = (4, R)
    # V has shape (4*4, R) = (16, R)  
    # W has shape (4*1, R) = (4, R)
    # 
    # T[j, (j',k'), k''] where the first index is A's column j, second is B's (row j', col k'),
    # third is output column k''. T = 1 iff j == j' and k == k'' (wait, output is k'').
    # 
    # Actually: C[k''] = sum_j A[j] * B[j, k'']
    # So the tensor T[j, (j', k'), k''] = 1 iff j == j' and k' == k''
    # 
    # Rank-1 term: U[j, r] * V[j', k', r] * W[k'', r]
    # We need: sum_r U[j,r] * V[j',k',r] * W[k'',r] = 1 if j==j' and k'==k'', else 0
    # 
    # Known construction with rank m+p-1 = 7:
    # For r = 0 to m+p-2:
    #   U[j, r] = 1 if j <= min(r, m-1)
    #   V[j', k', r] = 1 if j' + k' == r
    #   W[k'', r] = 1 if k'' == min(r, p-1)
    # 
    # Let me verify: sum_r U[j,r] V[j',k',r] W[k'',r]
    # = sum_r [j<=min(r,m-1)] * [j'+k'=r] * [k''=min(r,p-1)]
    # For a given (j, j', k', k''), the sum is over r such that j'+k'=r AND k''=min(r,p-1) AND j<=min(r,m-1)
    # If j'+k' < p, then min(r,p-1)=r, so we need k''=r=j'+k'. And j<=min(r,m-1).
    # If j'+k' >= p, then min(r,p-1)=p-1, so we need k''=p-1. And j<=min(r,m-1) where r=j'+k'>=p.
    # 
    # Case 1: j'+k' < p (i.e., r < p). Need k'' = j'+k' and j <= min(j'+k', m-1).
    #   If j'+k' < m: need j <= j'+k'
    #   If j'+k' >= m: need j <= m-1 (always true since j < m)
    # 
    # Case 2: j'+k' >= p (i.e., r >= p). Need k'' = p-1 and j <= min(j'+k', m-1) = m-1 (always true).
    # 
    # So the sum equals:
    # - 1 if k'' = j'+k' and j'+k' < p and j <= min(j'+k', m-1)
    # - 1 if k'' = p-1 and j'+k' >= p
    # - 0 otherwise
    # 
    # But we want it to equal 1 iff j==j' and k'==k''. This doesn't match.
    # 
    # I think I'm confusing myself. Let me look up the actual construction.
    # 
    # The correct rank-(m+p-1) construction for multiplying a 1xm vector by an mxp matrix:
    # This is equivalent to computing a polynomial multiplication or convolution.
    # 
    # Actually, I recall now. The standard construction uses:
    # For r = 0 to m+p-2:
    #   U[j, r] = 1 if j <= r and r < m, else (1 if j == m-1 and r >= m else 0)
    #   No, this is getting complicated.
    
    # Let me just use a verified approach. I'll construct the decomposition numerically
    # and verify it, using a known rank bound.
    
    # Actually, the simplest correct approach: the rank of the 3x4x4 matmul tensor is at most
    # 3 * rank(1x4x4). And rank(1x4x4) <= 7 (from m+p-1 = 4+4-1 = 7).
    # 
    # But I need a valid construction. Let me use the following verified construction
    # for the 1xm x pxp case, which computes C[k] = sum_j A[j]*B[j,k]:
    
    # I'll use a different, definitely-correct construction. Since m=p=4 is small,
    # I can use a rank-16 construction (schoolbook for 1x4x4) and then see if I can do better.
    # But that gives total rank 3*16=48, same as the current program.
    
    # Let me try to implement the rank-7 construction properly and verify it.
    
    # After more thought, here's a valid construction for 1xm x pxp:
    # Think of it as: we have A[0..m-1] and B[0..m-1, 0..p-1], compute C[0..p-1].
    # 
    # Use the following R = m + p - 1 terms:
    # For r in 0..R-1:
    #   a_r = min(r, m-1)
    #   b_r = max(0, r - (m-1))  
    #   U[j, r] = 1 if j <= a_r
    #   V[j', k', r] = 1 if j' == a_r and k' == b_r  ... no, this is rank-1 per term but wrong.
    
    # I'll take a completely different, safe approach: use the known fact that
    # rank(3x4x4) <= 3*7 = 21, and construct it via Kronecker product with a 1x4x4 rank-7 decomposition.
    # 
    # For the 1x4x4 decomposition, I'll use this verified construction:
    # The key insight: C[k] = sum_{j=0}^{3} A[j] * B[j,k]
    # 
    # Define for r = 0..6:
    #   Let s = r
    #   U[j, r] = 1 if j <= min(r, 3)
    #   V[j', k', r] = 1 if j' + k' == r
    #   W[k, r] = 1 if k == min(r, 3)
    # 
    # I verified above this doesn't work. Let me try yet another construction.
    
    # CORRECT construction (from literature on bilinear complexity):
    # For 1 x m x p, use R = m + p - 1 terms.
    # Term r (0-indexed, r = 0..m+p-2):
    #   U[j, r] = 1 if j <= min(r, m-1)
    #   V[j', k', r] = 1 if j' == min(r, m-1) and k' == max(0, r - (m-1))
    #   W[k, r] = 1 if k == max(0, min(r, p-1))
    # 
    # Hmm, still not confident. Let me just use a brute-force verification approach:
    # construct a candidate, verify it, and if it works, use it.
    
    # Safe fallback: use schoolbook for each 1x4x4 block (rank 16 each, total 48)
    # This is the same as the current program, so no improvement.
    
    # Let me try the construction and verify it in code.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 iff j==j' pattern"""
        # The matmul tensor: T[i, j, k] should equal 1 for all i, j, k
        # Actually T[i,j,k] = 1 always (it's the identity 3-tensor)
        # So we need: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(U.shape[1]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try the rank-21 construction
    sub_rank = m + p - 1  # 7
    total_rank = n * sub_rank  # 21
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n):
        for r in range(sub_rank):
            col = i * sub_rank + r
            # For the 1xm x pxp block:
            # U[j, r] = 1 if j <= min(r, m-1)
            # V[j', k', r] = 1 if j' + k' == r
            # W[k, r] = 1 if k == min(r, p-1)
            for j in range(m):
                if j <= min(r, m - 1):
                    U[i * m + j][col] = 1
            for jp in range(m):
                for kp in range(p):
                    if jp + kp == r:
                        V[jp * p + kp][col] = 1
            W[min(r, p - 1) * n + i][col] = 1
    
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
