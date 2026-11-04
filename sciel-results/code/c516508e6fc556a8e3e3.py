"""Tensor decomposition for 4x4 * 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where:
      u: length 4, indexed [i*2+j]
      v: length 4, indexed [j*2+k]
      w: length 4, indexed [k*2+i]
    """
    terms = []
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    u1 = [1, 0, 0, 1]; v1 = [1, 0, 0, 1]; w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # M2 = (A10+A11)B00 -> C10
    u2 = [0, 0, 1, 1]; v2 = [1, 0, 0, 0]; w2 = [0, 0, 1, 0]
    terms.append((u2, v2, w2))
    
    # M3 = A00(B10-B11) -> C01
    u3 = [1, 0, 0, 0]; v3 = [0, 0, 1, -1]; w3 = [0, 1, 0, 0]
    terms.append((u3, v3, w3))
    
    # M4 = A11(B00-B01) -> C10
    u4 = [0, 0, 0, 1]; v4 = [1, -1, 0, 0]; w4 = [0, 0, 1, 0]
    terms.append((u4, v4, w4))
    
    # M5 = (A00+A01)B11 -> C01
    u5 = [1, 1, 0, 0]; v5 = [0, 0, 0, 1]; w5 = [0, 1, 0, 0]
    terms.append((u5, v5, w5))
    
    # M6 = (A10-A00)(B00+B01) -> C10, C11
    u6 = [ -1, 0, 1, 0]; v6 = [1, 1, 0, 0]; w6 = [1, 0, 1, 0]
    terms.append((u6, v6, w6))
    
    # M7 = (A01-A11)(B10+B11) -> C01, C11
    u7 = [0, 1, 0, -1]; v7 = [0, 0, 1, 1]; w7 = [1, 0, 0, 1]
    terms.append((u7, v7, w7))
    
    return terms

def _construct_strassen_block(n=4, m=4, p=5):
    """
    Construct a decomposition using Strassen for 2x2x2 blocks where possible.
    For 4x4x5, we can decompose into blocks of size 2x2x2 and 2x2x1 (or similar).
    
    The tensor T[i,j,k] = 1 for all i,j,k.
    We can split indices into blocks of size 2.
    i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3,4}.
    
    Let's split i into {0,1} and {2,3}.
    Let's split j into {0,1} and {2,3}.
    k has 5 elements, so we can't split it evenly into 2x2 blocks.
    We can split k into {0,1}, {2,3}, {4}.
    
    This gives 2 * 2 * 3 = 12 blocks? No, the tensor is a single sum over j.
    C[i,k] = sum_j A[i,j] B[j,k].
    
    If we split j into j1 in {0,1} and j2 in {2,3}:
    C[i,k] = sum_{j1} A[i,j1] B[j1,k] + sum_{j2} A[i,j2] B[j2,k].
    
    This is the sum of two tensors:
    T1[i, j1, k] = 1 for i in {0..3}, j1 in {0,1}, k in {0..4}.
    T2[i, j2, k] = 1 for i in {0..3}, j2 in {2,3}, k in {0..4}.
    
    T1 is a 4x2x5 tensor. T2 is a 4x2x5 tensor.
    The rank of T1 + T2 is at most rank(T1) + rank(T2).
    
    Can we decompose T1 (4x2x5) efficiently?
    T1[i, j, k] = 1.
    This is the matmul tensor for A (4x2) * B (2x5) -> C (4x5).
    The schoolbook rank is 4*2*5 = 40.
    Can we do better?
    The border rank of 4x2x5 is at least max(4*2, 2*5, 4*5) = max(8, 10, 20) = 20?
    Actually, for A (m x n) * B (n x p), the rank is at least max(mn, np, mp) ? No.
    For 2x2x2, rank is 7. Schoolbook is 8.
    For 4x2x5, is there a known low rank?
    
    Let's try a different blocking.
    What if we use the 2x2x2 Strassen decomposition for the "core" and handle the rest?
    
    Let's look at the structure of the current best (schoolbook 80).
    Can we merge terms?
    
    Let's try a specific construction for 4x4x5.
    Consider the blocks of A: A00, A01, A10, A11 (each 2x2).
    B: B00, B01, B10, B11, B20, B21, B30, B31? No, B is 4x5.
    B rows 0,1,2,3. Cols 0,1,2,3,4.
    
    Let's split B columns into B_left (4x4) and B_right (4x1).
    C = A * B_left + A * B_right.
    A * B_left is 4x4x4 matmul.
    A * B_right is 4x4x1 matmul (matrix-vector).
    
    Rank of 4x4x4 matmul tensor?
    It is the sum of 4 disjoint 2x2x2 tensors?
    T[i,j,k] = 1 for i,j,k in 0..3.
    Split i,j,k into {0,1} and {2,3}.
    T = T000 + T001 + T010 + T011 + T100 + T101 + T110 + T111?
    No, the indices are not independent in that way.
    T[i,j,k] = 1.
    If i in {0,1}, j in {0,1}, k in {0,1}, this is a 2x2x2 block.
    There are 2*2*2 = 8 such blocks in 4x4x4.
    Since the supports are disjoint, the rank is the sum of the ranks.
    Rank(4x4x4) = 8 * Rank(2x2x2) = 8 * 7 = 56.
    
    So we can get a rank 56 decomposition for the 4x4x4 part.
    
    Now for the 4x4x1 part (A * B_col4).
    This is a 4x4x1 tensor.
    T[i,j,0] = 1.
    This is just the tensor for matrix-vector multiplication.
    Rank of 4x4x1 tensor?
    It is equivalent to the identity tensor of size 4?
    T[i,j] = 1.
    This is the all-ones matrix.
    The rank of the tensor T_{ij} is the matrix rank of the matrix T.
    The matrix of all ones 4x4 has rank 1.
    So we can decompose it into 1 term?
    U[i,j] = 1, V[j,0] = 1, W[0,i] = 1?
    Sum_r U[i,j,r] V[j,0,r] W[0,i,r] = U[i,j,0] V[j,0,0] W[0,i,0].
    If U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1, then sum is 1.
    So rank 1.
    
    Total rank = 56 + 1 = 57.
    
    Let's verify this logic.
    The tensor for 4x4 * 4x5 is T[i,j,k] = 1.
    We split k into k in {0,1,2,3} and k=4.
    T = T_{k<4} + T_{k=4}.
    T_{k<4} is the 4x4x4 matmul tensor.
    T_{k=4} is the 4x4x1 matmul tensor.
    
    Since the supports in k are disjoint, Rank(T) = Rank(T_{k<4}) + Rank(T_{k=4}).
    
    Rank(T_{k<4}) = Rank(4x4x4 matmul).
    As derived, 4x4x4 matmul tensor is the direct sum of 8 2x2x2 matmul tensors.
    Rank = 8 * 7 = 56.
    
    Rank(T_{k=4}) = Rank(4x4x1 matmul).
    T[i,j,0] = 1.
    This tensor is rank 1. (U[i,j]=1, V[j,0]=1, W[0,i]=1).
    
    So total rank is 57.
    
    This is a significant improvement over 80.
    
    Let's construct this.
    
    1. Construct 8 blocks of 2x2x2 Strassen terms for the 4x4x4 part.
       Blocks are defined by i_block in {0,1}, j_block in {0,1}, k_block in {0,1}.
       i = 2*i_block + li, j = 2*j_block + lj, k = 2*k_block + lk.
       
       For each block, we add 7 terms.
       Total 56 terms.
       
    2. Add 1 term for the k=4 part.
       U[i, j, 56] = 1 for all i, j.
       V[j, 4, 56] = 1 for all j. (Note: V index is j*p+k. Here k=4. So V[j*5+4, 56] = 1).
       W[4, i, 56] = 1 for all i. (Note: W index is k*n+i. Here k=4. So W[4*4+i, 56] = 1).
       
    Let's double check the indices.
    U shape: (n*m, rank) = (16, 57).
    V shape: (m*p, rank) = (20, 57).
    W shape: (p*n, rank) = (20, 57).
    
    For the 4x4x4 part (k=0..3):
    For each (ib, jb, kb) in 0..1:
      For each Strassen term t in 0..6:
        u, v, w = strassen_terms[t]
        r = (ib*2 + jb)*2 + kb)*7 + t  # Wait, block index needs to be unique.
        Block index b = ib*4 + jb*2 + kb.
        r = b*7 + t.
        
        For local indices li, lj, lk in 0..1:
          gi = 2*ib + li
          gj = 2*jb + lj
          gk = 2*kb + lk
          
          U[gi*m + gj, r] += u[li*2 + lj]
          V[gj*p + gk, r] += v[lj*2 + lk]
          W[gk*n + gi, r] += w[lk*2 + li]
          
    For the k=4 part:
      r = 56.
      For i in 0..3, j in 0..3:
        U[i*4 + j, 56] = 1
      For j in 0..3:
        V[j*5 + 4, 56] = 1
      For i in 0..3:
        W[4*4 + i, 56] = 1
        
    Let's check if this is valid.
    Sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    For k in 0..3:
      Only terms with gk=k contribute.
      gk = 2*kb + lk.
      So kb = k//2, lk = k%2.
      The sum over r for this k is the sum over blocks with this kb, and all jb, ib.
      Wait, the tensor is T[i,j,k] = 1.
      The decomposition sums to 1 for each (i,j,k).
      
      Let's check one specific (i,j,k).
      Let i=0, j=0, k=0.
      ib=0, jb=0, kb=0. li=0, lj=0, lk=0.
      Only the block (0,0,0) contributes to U[0,0,r], V[0,0,r], W[0,0,r].
      Actually, U[0,0,r] is non-zero only for r in block (0,0,0).
      V[0,0,r] is non-zero only for r in block (0,0,0).
      W[0,0,r] is non-zero only for r in block (0,0,0).
      
      So the sum is sum_{t=0..6} u_t[0,0] v_t[0,0] w_t[0,0].
      This is the reconstruction of the 2x2x2 tensor at (0,0,0).
      The Strassen decomposition is exact, so this sum is 1.
      
      Now consider i=0, j=0, k=4.
      k=4 is handled by the last term r=56.
      U[0,0,56] = 1.
      V[0,4,56] = 1.
      W[4,0,56] = 1.
      Sum = 1*1*1 = 1.
      Other terms have W[k,i,r] = 0 for k=4?
      For r < 56, gk is in 0..3. So W[4,0,r] = 0.
      So only r=56 contributes. Sum is 1.
      
      What about i=0, j=2, k=0?
      i=0 -> ib=0, li=0.
      j=2 -> jb=1, lj=0.
      k=0 -> kb=0, lk=0.
      U[0,2,r] is non-zero for blocks with ib=0, jb=1.
      V[2,0,r] is non-zero for blocks with jb=1, kb=0.
      W[0,0,r] is non-zero for blocks with kb=0, ib=0.
      
      The intersection of blocks is (ib=0, jb=1, kb=0).
      So only terms from this block contribute.
      The local indices are li=0, lj=0, lk=0.
      Sum is u_t[0,0] v_t[0,0] w_t[0,0] for t in this block.
      This is the value of the 2x2x2 tensor at local (0,0,0), which is 1.
      
      This seems correct.
      
    So the rank is 57.
    
    Can we do better?
    The 4x4x1 part has rank 1.
    The 4x4x4 part has rank 56.
    Is there a way to share terms between the 4x4x4 and 4x4x1 parts?
    The supports in k are disjoint (0..3 vs 4), so no.
    
    Is the rank of 4x4x4 really 56?
    The 4x4x4 matmul tensor is the direct sum of 8 2x2x2 tensors.
    The rank of a direct sum is the sum of the ranks.
    Rank(2x2x2) = 7.
    So Rank(4x4x4) = 56.
    This is a known result.
    
    So 57 is a solid lower bound for this construction.
    The schoolbook rank is 80.
    57 is a significant improvement.
    
    Let's implement this.
    """
    strassen_terms = _strassen_2x2x2_terms()
    rank = 57
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 4x4x4 part (k=0..3)
    # 8 blocks of 2x2x2
    block_idx = 0
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                for t in range(7):
                    r = block_idx * 7 + t
                    u, v, w = strassen_terms[t]
                    for li in range(2):
                        for lj in range(2):
                            for lk in range(2):
                                gi = 2 * ib + li
                                gj = 2 * jb + lj
                                gk = 2 * kb + lk
                                U[gi * m + gj, r] += u[li * 2 + lj]
                                V[gj * p + gk, r] += v[lj * 2 + lk]
                                W[gk * n + gi, r] += w[lk * 2 + li]
                block_idx += 1
    
    # Part 2: 4x4x1 part (k=4)
    r = 56
    for i in range(n):
        for j in range(m):
            U[i * m + j, r] = 1
    for j in range(m):
        V[j * p + 4, r] = 1
    for i in range(n):
        W[4 * n + i, r] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_remove_terms(U, V, W, n, m, p, max_iterations=100, time_limit=30):
    """
    Try to remove redundant terms from a decomposition.
    """
    start_time = time.time()
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _is_valid(U_mat, V_mat, W_mat, r):
        if r <= 0:
            return False
        U3 = U_mat[:, :r].reshape(n, m, -1)
        V3 = V_mat[:, :r].reshape(m, p, -1)
        W3 = W_mat[:, :r].reshape(p, n, -1)
        M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
        return np.allclose(M_decomp, 1.0, atol=1e-9)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        # Try removing each term
        for t in range(rank):
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, t] = 0
            V_test[:, t] = 0
            W_test[:, t] = 0
            
            if _is_valid(U_test, V_test, W_test, rank):
                U = np.delete(U, t, axis=1)
                V = np.delete(V, t, axis=1)
                W = np.delete(W, t, axis=1)
                rank -= 1
                found = True
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if not found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}, best_rank

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-embedded approach
    try:
        strassen_result = _construct_strassen_block(n, m, p)
        rank_s = len(strassen_result["U"][0]) if strassen_result["U"] else 0
        if rank_s < best_rank:
            if _validate(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                best_result = strassen_result
                best_rank = rank_s
    except Exception:
        pass
    
    # Try to remove redundant terms
    try:
        U = np.array(best_result["U"], dtype=float)
        V = np.array(best_result["V"], dtype=float)
        W = np.array(best_result["W"], dtype=float)
        greedy_result, greedy_rank = _try_remove_terms(U, V, W, n, m, p, max_iterations=50, time_limit=20)
        if greedy_rank < best_rank:
            if _validate(greedy_result["U"], greedy_result["V"], greedy_result["W"], n, m, p):
                best_result = greedy_result
                best_rank = greedy_rank
    except Exception:
        pass
    
    return best_result

# EVOLVE-BLOCK-END
