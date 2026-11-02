"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

_DEADLINE = time.time() + 115.0

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: for all i,j,k sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _construct_strassen_block_decomposition():
    """
    Construct a rank-66 decomposition for 3x4x6 matmul by embedding
    the rank-11 decomposition of M_{3,2,2} (Strassen on top 2x2 + schoolbook for row 3)
    into each (row-block, col-block) pair.
    
    The 3x4x6 matmul can be viewed as 3 row-blocks (each of height 2, except last of height 1... 
    no, let's think differently).
    
    Actually, let's partition columns into 3 groups of 2, and rows into 2 groups of 2 (with 1 extra).
    Wait: 4 columns = 2+2, 6 rows... no.
    
    Let me re-think: n=3 (output rows), m=4 (shared dim), p=6 (output cols).
    
    Partition: m=4 into two blocks of 2: j in {0,1} and j in {2,3}.
    p=6 into three blocks of 2: k in {0,1}, {2,3}, {4,5}.
    n=3: we handle all 3 output rows together with each 2x2 sub-multiplication.
    
    For each pair (block of 2 columns in B, block of 2 rows in A), we compute a 3x2 output.
    The sub-problem is: C_{3x2} = A_{3x2} * B_{2x2}
    This is M_{3,2,2}.
    
    We have 2 blocks in A (j=0,1 and j=2,3) and 3 blocks in B (k=0,1; 2,3; 4,5).
    Total: 2 * 3 = 6 sub-problems, each of type M_{3,2,2}.
    
    For M_{3,2,2}: C = A * B where A is 3x2, B is 2x2, C is 3x2.
    A = [[a, b], [c, d], [e, f]]
    B = [[g, h], [i, j]]
    C = [[ag+bh, ah+bj], [cg+dh, ch+dj], [eg+fh, eh+fj]]
    
    Using Strassen for the top 2x2 block (rows 0,1 of C) plus schoolbook for row 2:
    Strassen (7 products):
    P1 = (a+b)(g+i)
    P2 = b(g+i)  
    P3 = a(h-i)
    P4 = (c+d)(g+h)
    P5 = c(h-j)
    P6 = (a+c)(g+i)  -- wait, this isn't standard. Let me use the actual 7.
    
    Standard Strassen for 2x2:
    C00 = ag+bh, C01 = ah+bj, C10 = cg+dh, C11 = ch+dj
    M1 = (a+b)(g+i)
    M2 = (a-b)(i+j)  -- no, that's not right either.
    
    Let me just use: the 2x2 matmul has rank 7. The specific 7 rank-1 terms are:
    For C00 = ag + bh:
    For C01 = ah + bj:
    For C10 = cg + dh:
    For C11 = ch + dj:
    
    One known rank-7 decomposition:
    M1 = (a+b)(g+i)
    M2 = b(g+i)
    M3 = a(h-i)
    M4 = (c+d)(g+h)
    M5 = c(h-j)
    M6 = (a+c)(g+i)
    M7 = ... 
    
    Actually, let me just use a different approach. The rank of M_{2,2,2} is 7.
    For M_{3,2,2}, we can take the rank-7 for the top 2 rows plus 4 schoolbook terms for row 3.
    Total rank = 7 + 4 = 11.
    
    But actually, we can do better. Let me just use schoolbook for the whole 3x2x2 which is 12,
    and see if that's already better than 72. 6 * 12 = 72. Same as schoolbook.
    
    Hmm. Let me think differently.
    
    For M_{3,2,2}, the tensor is:
    T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1}, k in {0,1}
    with the bilinear form sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_j (A[0,j]B[j,0] + A[0,j]B[j,1] + A[1,j]B[j,0] + A[1,j]B[j,1] + A[2,j]B[j,0] + A[2,j]B[j,1])
    
    Wait no. The matmul tensor M_{n,m,p} has T[i,j,k] = 1 for all valid indices, and the
    bilinear form is <M, A⊗B> = sum_{i,j,k} A[i,j] B[j,k] = sum_i sum_k (AB)[i,k]... 
    no that's summing all entries.
    
    Actually I think the convention is: the decomposition must satisfy
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    So for M_{3,2,2}: n=3, m=2, p=2.
    U has 3*2=6 rows, V has 2*2=4 rows, W has 2*3=6 rows.
    The condition is: for all i in {0,1,2}, j in {0,1}, k in {0,1}:
      sum_r U[2i+j, r] * V[2j+k, r] * W[2k+i, r] = 1
    
    Schoolbook rank = 3*2*2 = 12.
    
    Can we do rank 11? That would require showing that one linear combination is dependent.
    
    For now, let's just use the block approach with rank-12 schoolbook per block: 6*12 = 72.
    That's the same as full schoolbook. Not an improvement.
    
    Let me try a different partition. What if I use rank-7 Strassen for the 2x2 sub-blocks?
    
    Actually, I realize the issue. The decomposition condition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i, j, k.
    
    This means the tensor is ALL ONES (n*m*p ones). The rank of the all-ones tensor of 
    size n*m*p is n*m*p (each entry needs its own rank-1 term in the worst case... no wait).
    
    Actually, the all-ones tensor of size a x b x c has tensor rank max(a,b,c) in general... 
    no. The tensor rank of the all-ones tensor J_{a,b,c} is not simply max. 
    
    For J_{2,2,2}, the rank is 7 (this is the famous result - Strassen).
    For J_{3,2,2}, what's the rank? 
    
    Actually wait. I was confusing myself. Let me re-read the problem.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix."
    
    The matrix multiplication tensor M_{3,4,6} is defined by:
    (A*B)[i,k] = sum_j A[i,j] B[j,k]
    
    This is a trilinear form: sum_{i,j,k} A[i,j] B[j,k] C'[k,i] where C' is the transpose of the output.
    Or equivalently, the tensor T has T[i,j,k] such that the bilinear map (A,B) -> <T, A⊗B> gives the matmul.
    
    In the standard tensor decomposition framework:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    
    And the condition is: for all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    Where T[i,j,k] = 1 for all i, j, k (since matmul just sums over j, and the "coefficient" is 1).
    
    Wait, that's not right either. Let me think about this more carefully.
    
    The matrix multiplication can be expressed as:
    (AB)[i,k] = sum_j A[i,j] B[j,k]
    
    This is a bilinear map R^{n*m} × R^{m*p} -> R^{n*p}.
    The associated trilinear form (using the standard basis) is:
    T(A, B, C) = sum_{i,j,k} A[i,j] B[j,k] C[k,i]  (if we identify output with its transpose)
    
    Hmm, but the W index is k*n+i, suggesting W is indexed by (k,i) pairs.
    
    So the tensor T has entries T[i,j,k] = 1 for all valid i, j, k.
    The decomposition T = sum_r U_r ⊗ V_r ⊗ W_r means:
    T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 for all i,j,k.
    
    So yes, it's the all-ones tensor! And we need its tensor rank.
    
    The tensor rank of J_{3,4,6} (all-ones 3x4x6 tensor).
    
    Known results:
    - J_{2,2,2} has rank 7
    - J_{2,2,n} has rank 2n-1 (for n >= 2)
    - J_{n,m,p} for small dimensions...
    
    For J_{3,4,6}:
    Lower bound: rank >= max(3,4,6) = 6 (trivial), but actually rank >= n+m+p-2 = 3+4+6-2 = 11 (Barg-Motzkin-Szekeres bound for the all-ones tensor? No, that's for a different tensor.)
    
    Actually, the all-ones tensor J_{a,b,c} has tensor rank:
    - If a=b=c=2: rank 7
    - Generally, rank(J_{a,b,c}) <= min(ab, bc, ac) (schoolbook)
    
    For our case, schoolbook = 3*4*6 = 72.
    
    Can we do better? Let's think about it as: we need to write the all-ones 3x4x6 tensor as a sum of rank-1 tensors.
    
    A rank-1 tensor is u ⊗ v ⊗ w where u ∈ R^3... no wait. U is n*m = 12 dimensional, V is m*p = 24 dimensional, W is p*n = 18 dimensional.
    
    Hmm, this is a 12 x 24 x 18 tensor that is all-ones.
    
    The tensor rank of an all-ones a x b x c tensor is known to be:
    - rank(J_{a,b,c}) = min(a,b,c) if one of them is 1
    - rank(J_{2,2,2}) = 7
    - For larger dimensions, it's not always known exactly.
    
    Actually, I recall that the all-ones tensor J_{a,b,c} has rank at most a+b+c-2 when one dimension is 2.
    More generally, there are results but they're complex.
    
    For a practical approach, let me try to use the Strassen-like decomposition.
    
    Key insight: J_{2,2,2} has rank 7. This means we can decompose the 2x2x2 all-ones tensor with 7 terms.
    
    For J_{3,4,6}, we can try to embed this.
    
    Let me think of a constructive approach:
    
    Partition the indices:
    - i ∈ {0,1,2} (3 values)
    - j ∈ {0,1,2,3} (4 values)  
    - k ∈ {0,1,2,3,4,5} (6 values)
    
    We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    Approach: Use the rank-7 decomposition of J_{2,2,2} as a building block.
    
    For J_{2,2,2}, the 7 rank-1 terms (in terms of indices (i',j',k') where each is 0 or 1):
    
    The standard Strassen decomposition of J_{2,2,2}:
    Let me denote the 7 terms by their (i',j',k') patterns.
    
    Actually, for the all-ones 2x2x2 tensor, the rank-7 decomposition is:
    T[i,j,k] = sum_{r=1}^{7} u_r[i] v_r[j] w_r[k] = 1 for all i,j,k ∈ {0,1}
    
    One such decomposition (from Strassen's algorithm):
    Term 1: u=(1,1), v=(1,0), w=(1,1)  -> contributes to (i,j,k) where... 
    
    Hmm, let me think of it differently. In Strassen's algorithm:
    C00 = P1 - P2 - P4 + P7
    C01 = P3 + P7
    C10 = P1 + P5
    C11 = P2 - P5 - P6 + P7
    
    where:
    P1 = (a+b)(e+f)
    P2 = (a+b)(g+i)  -- wait, this isn't standard notation.
    
    Let me use: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    P1 = (a+b)(e+f)
    P2 = b(e+f)
    P3 = (c+d)(e+g)
    P4 = d(e+g)
    P5 = (a+c)(h-g)
    P6 = (b+d)(f-h)
    P7 = (a-b)(e+h)  -- no this isn't right.
    
    The actual Strassen:
    P1 = (a+b)(e+f)
    P2 = b(e+f)
    P3 = (c+d)(e+g)
    P4 = d(e+g)
    P5 = (a+c)(h-g)
    P6 = (b+d)(f-h)
    P7 = (a-b)(e+h)
    
    C00 = P1 - P2 - P4 + P7  -- let me verify: (a+b)(e+f) - b(e+f) - d(e+g) + (a-b)(e+h)
    = (a+b-e... let me expand:
    = ae+af+be+bf - be-bf - de-dg + ae+ah-be-bh
    = 2ae + af + be + bf - be - bf - de - dg + ae + ah - be - bh
    Hmm this is getting messy. Let me just use a known explicit decomposition.
    
    For the 2x2x2 all-ones tensor, one rank-7 decomposition is:
    
    r1: U=[1,1], V=[1,0], W=[1,0]  -> T[0,0,0]+=1, T[1,0,0]+=1
    r2: U=[1,1], V=[1,0], W=[0,1]  -> T[0,0,1]+=1, T[1,0,1]+=1
    r3: U=[1,1], V=[0,1], W=[1,0]  -> T[0,1,0]+=1, T[1,1,0]+=1
    r4: U=[1,1], V=[0,1], W=[0,1]  -> T[0,1,1]+=1, T[1,1,1]+=1
    r5: U=[1,-1], V=[1,1], W=[1,0]  -> T[0,0,0]+=1, T[0,1,0]+=1, T[1,0,0]+=-1, T[1,1,0]+=-1
    r6: U=[1,-1], V=[1,1], W=[0,1]  -> T[0,0,1]+=1, T[0,1,1]+=1, T[1,0,1]+=-1, T[1,1,1]+=-1
    r7: U=[1,-1], V=[1,-1], W=[1,1]  -> ... 
    
    This is getting complicated. Let me just use a numerical optimization approach with a better initialization.
    
    Actually, let me just try to find the rank numerically using alternating least squares with multiple random restarts, starting from rank 20 and going down.

    n, m, p = 3, 4, 6
    U_dim, V_dim, W_dim = n*m, m*p, p*n  # 12, 24, 18
    
    for target_rank in range(20, 73):
        # Try to find a decomposition of this rank
        best_err = float('inf')
        best_result = None
        
        for restart in range(5):
            if time.time() > _DEADLINE - 5:
                break
            
            np.random.seed(restart * 1000 + target_rank)
            U = np.random.randn(U_dim, target_rank) * 0.1
            V = np.random.randn(V_dim, target_rank) * 0.1
            W = np.random.randn(W_dim, target_rank) * 0.1
            
            for it in range(200):
                if time.time() > _DEADLINE - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                    A_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                    B_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        B_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                    C_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        C_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C_mat, np.ones(m), rcond=None)[0]
                    except:
                        pass
                
                # Check error
                if it % 20 == 0:
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                                err = max(err, abs(s - 1.0))
                    if err < 1e-10:
                        return (U.tolist(), V.tolist(), W.tolist())
        
        # If we got here, rank target_rank didn't work, try higher
        
    return None
"""

def _construct_strassen_block_decomposition():
    """
    Construct a decomposition for 3x4x6 matmul using block structure.
    
    Partition: j into 2 blocks of 2, k into 3 blocks of 2.
    For each (j-block, k-block), we have a 3x2x2 sub-problem.
    Use rank-12 schoolbook for each: total 6*12 = 72.
    
    To improve, use rank-11 for M_{3,2,2} if possible.
    For now, just use the schoolbook per block (same as full schoolbook).
    
    Actually, let's try a different approach: use the fact that J_{2,2,2} has rank 7.
    We can write J_{3,4,6} using this building block.
    
    Key idea: J_{a,b,c} can be built from J_{2,2,2} blocks with some overlap.
    
    For now, let's just use the numerical ALS approach with better parameters.
    """
    n, m, p = 3, 4, 6
    U_dim, V_dim, W_dim = n * m, m * p, p * n
    
    def _als_decompose(target_rank, max_iter=300, n_restarts=3):
        for restart in range(n_restarts):
            if time.time() > _DEADLINE - 5:
                return None
            
            np.random.seed(42 + restart * 100 + target_rank)
            U = np.random.randn(U_dim, target_rank) * 0.05
            V = np.random.randn(V_dim, target_rank) * 0.05
            W = np.random.randn(W_dim, target_rank) * 0.05
            
            for it in range(max_iter):
                if time.time() > _DEADLINE - 5:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        B_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            B_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            C_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(C_mat, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check error every 10 iterations
                if it % 10 == 0:
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                err = max(err, abs(s - 1.0))
                    if err < 1e-10:
                        return (U.tolist(), V.tolist(), W.tolist())
        
        return None
    
    # Try ranks from 20 up to 72
    for target_rank in range(20, 73):
        if time.time() > _DEADLINE - 10:
            break
        result = _als_decompose(target_rank)
        if result is not None:
            U_list, V_list, W_list = result
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try the structured/numerical approach
    result = _construct_strassen_block_decomposition()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
