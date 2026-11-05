"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

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
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """Return rank-1 vectors for 2x2 * 2x2 matrix multiplication.
    u vectors are in R^4 (indices i*m+j with m=2)
    v vectors are in R^4 (indices j*p+k with p=2)
    w vectors are in R^4 (indices k*n+i with n=2)
    """
    # Strassen 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A11-A01)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Index conventions:
    # A is 2x2, index by (i,j) -> i*2+j
    # B is 2x2, index by (j,k) -> j*2+k
    # C is 2x2, index by (i',k') in trace convention: W[k*n+i] = W[k*2+i]
    # So C[i,k] has index k*2+i
    
    # A00 = A[0,0] index 0, A01 = A[0,1] index 1
    # A10 = A[1,0] index 2, A11 = A[1,1] index 3
    # B00 = B[0,0] index 0, B01 = B[0,1] index 1
    # B10 = B[1,0] index 2, B11 = B[1,1] index 3
    # C[i,k] index k*2+i:
    # C[0,0] index 0, C[0,1] index 1
    # C[1,0] index 2, C[1,1] index 3
    
    U_vecs = []
    V_vecs = []
    W_vecs = []
    
    # M1 = (A00+A11)(B00+B11)
    u1 = np.zeros(4); u1[0] = 1; u1[3] = 1
    v1 = np.zeros(4); v1[0] = 1; v1[3] = 1
    w1 = np.zeros(4); w1[0] = 1; w1[3] = 1
    U_vecs.append(u1); V_vecs.append(v1); W_vecs.append(w1)
    
    # M2 = (A10+A11)B00
    u2 = np.zeros(4); u2[2] = 1; u2[3] = 1
    v2 = np.zeros(4); v2[0] = 1
    w2 = np.zeros(4); w2[2] = 1; w2[3] = 1
    U_vecs.append(u2); V_vecs.append(v2); W_vecs.append(w2)
    
    # M3 = A00(B01-B11)
    u3 = np.zeros(4); u3[0] = 1
    v3 = np.zeros(4); v3[1] = 1; v3[3] = -1
    w3 = np.zeros(4); w3[1] = 1
    U_vecs.append(u3); V_vecs.append(v3); W_vecs.append(w3)
    
    # M4 = A11(B10-B00)
    u4 = np.zeros(4); u4[3] = 1
    v4 = np.zeros(4); v4[2] = 1; v4[0] = -1
    w4 = np.zeros(4); w4[2] = 1
    U_vecs.append(u4); V_vecs.append(v4); W_vecs.append(w4)
    
    # M5 = (A00+A10)B11
    u5 = np.zeros(4); u5[0] = 1; u5[2] = 1
    v5 = np.zeros(4); v5[3] = 1
    w5 = np.zeros(4); w5[1] = -1; w5[3] = -1
    U_vecs.append(u5); V_vecs.append(v5); W_vecs.append(w5)
    
    # M6 = (A01-A10)(B00+B10)
    u6 = np.zeros(4); u6[1] = 1; u6[2] = -1
    v6 = np.zeros(4); v6[0] = 1; v6[2] = 1
    w6 = np.zeros(4); w6[0] = 1
    U_vecs.append(u6); V_vecs.append(v6); W_vecs.append(w6)
    
    # M7 = (A11-A01)(B01+B11)
    u7 = np.zeros(4); u7[3] = 1; u7[1] = -1
    v7 = np.zeros(4); v7[1] = 1; v7[3] = 1
    w7 = np.zeros(4); w7[3] = 1
    U_vecs.append(u7); V_vecs.append(v7); W_vecs.append(w7)
    
    return U_vecs, V_vecs, W_vecs

def _verify_decomp(U, V, W, n=2, m=4, p=5):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta(i,i')*delta(j,j')*delta(k,k')
    for the matrix multiplication tensor T[(i,j),(j',k),(k',i')] = delta(i,k')*delta(j,j')*delta(k,i')... 
    
    Actually the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    T[(i,j),(j',k),(i',k')] = delta(j,j') * delta(i,i') * delta(k,k')
    
    Wait, let me think again. The multiplication tensor T satisfies:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So T[A[i,j], B[j,k], C[i,k]] = 1 if the indices match, 0 otherwise.
    More precisely, T[(i,j),(j',k),(i',k')] = delta(j,j') * delta(i,i') * delta(k,k')
    
    In our flattening:
    U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    V has shape (m*p, rank), indexed by (j,k) -> j*p+k
    W has shape (p*n, rank), indexed by (k,i) -> k*n+i  [trace convention W[k*n+i]]
    
    The decomposition says: for all (i,j),(j',k),(k',i'):
    sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta(j,j') * delta(i,i') * delta(k,k')
    
    Let me verify by building the full tensor and comparing.
    """
    rank = U.shape[1]
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = delta(j,j') * delta(i,i') * delta(k,k')
    # where i = a//m, j = a%m, j' = b//p, k = b%p, k' = c//n, i' = c%n
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Compute the decomposition tensor
    # D[a,b,c] = sum_c U[a,c] * V[b,c] * W[c_,c] where c_ is the third index
    # This is a tensor product contraction
    # D = U[:,:,None,None] * V[None,:,None,None] * W[None,None,:,None] contracted over last axis
    # More efficiently: D[a,b,c_] = sum_r U[a,r]*V[b,r]*W[c_,r]
    # = (U * V^T) outer ... no, let's do it as:
    # For each r, outer product of U[:,r], V[:,r], W[:,r]
    
    # Use einsum for clarity
    # D[a,b,c_] = sum_r U[a,r] * V[b,r] * W[c_,r]
    D = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            if j == j2 and i == i2 and k == k2:
                                a = i*m + j
                                b = j2*p + k
                                c_ = k2*n + i2
                                T_target[a, b, c_] = 1.0
    
    max_err = np.max(np.abs(D - T_target))
    return max_err

def _strassen_block(n, m, p, U_s, V_s, W_s, i_off, j_off, k_off):
    """Place a Strassen 2x2 block at offset (i_off, j_off, k_off).
    Returns lists of (U_vec, V_vec, W_vec) in the full coordinate system.
    """
    results = []
    for c in range(7):
        u_full = np.zeros(n*m)
        v_full = np.zeros(m*p)
        w_full = np.zeros(p*n)
        
        # Place u_s[c] (4 entries for 2x2 A block) at rows i_off*2..i_off*2+1, cols j_off*2..j_off*2+1
        for ii in range(2):
            for jj in range(2):
                idx_in_block = ii*2 + jj
                full_idx = (i_off + ii)*m + (j_off + jj)
                u_full[full_idx] = U_s[c][idx_in_block]
        
        # Place v_s[c] (4 entries for 2x2 B block) at rows j_off*2..j_off*2+1, cols k_off*2..k_off*2+1
        for jj in range(2):
            for kk in range(2):
                idx_in_block = jj*2 + kk
                full_idx = (j_off + jj)*p + (k_off + kk)
                v_full[full_idx] = V_s[c][idx_in_block]
        
        # Place w_s[c] (4 entries for 2x2 C block) at rows k_off*2..k_off*2+1, cols i_off*2..i_off*2+1
        for kk in range(2):
            for ii in range(2):
                idx_in_block = kk*2 + ii
                full_idx = (k_off + kk)*n + (i_off + ii)
                w_full[full_idx] = W_s[c][idx_in_block]
        
        results.append((u_full, v_full, w_full))
    
    return results

def _schoolbook_block(n, m, p, i_off, j_off, k_off, ni=2, nj=2, nk=2):
    """Place a schoolbook ni x nj * nj x nk block at offset (i_off, j_off, k_off)."""
    results = []
    for ii in range(ni):
        for jj in range(nj):
            for kk in range(nk):
                u_full = np.zeros(n*m)
                v_full = np.zeros(m*p)
                w_full = np.zeros(p*n)
                u_full[(i_off+ii)*m + (j_off+jj)] = 1.0
                v_full[(j_off+jj)*p + (k_off+kk)] = 1.0
                w_full[(k_off+kk)*n + (i_off+ii)] = 1.0
                results.append((u_full, v_full, w_full))
    return results

def _build_block_decomposition():
    """Build rank-38 decomposition using Strassen for 2x2*2x2 blocks and schoolbook for 2x2*2x1.
    
    Split j (size 4) into two blocks of 2: j_block 0 (j=0,1) and j_block 1 (j=2,3).
    For each j_block, split k (size 5) into 2+2+1: k_block 0 (k=0,1), k_block 1 (k=2,3), k_block 2 (k=4).
    
    For (j_block, k_block):
    - If nk=2: use Strassen (rank 7)
    - If nk=1: use schoolbook 2x2*2x1 (rank 4)
    
    Total: 2 * (7 + 7 + 4) = 36
    
    Wait, let me reconsider. The i dimension is size 2, which fits in one block.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    U_s, V_s, W_s = _strassen_2x2()
    
    all_U = []
    all_V = []
    all_W = []
    
    # For each j_block (0, 1) and k_block configuration
    for jb in range(2):  # j_block: 0 or 1
        j_off = jb * 2  # j offset: 0 or 2
        
        # k_block 0: k=0,1 (nk=2) -> Strassen
        # k_block 1: k=2,3 (nk=2) -> Strassen  
        # k_block 2: k=4 (nk=1) -> schoolbook 2x2 * 2x1
        
        # Strassen for (i=0:2, j=j_off:j_off+2, k=0:2)
        for (u, v, w) in _strassen_block(n, m, p, U_s, V_s, W_s, 0, j_off, 0):
            all_U.append(u)
            all_V.append(v)
            all_W.append(w)
        
        # Strassen for (i=0:2, j=j_off:j_off+2, k=2:4)
        for (u, v, w) in _strassen_block(n, m, p, U_s, V_s, W_s, 0, j_off, 2):
            all_U.append(u)
            all_V.append(v)
            all_W.append(w)
        
        # Schoolbook for (i=0:2, j=j_off:j_off+2, k=4:5)
        for (u, v, w) in _schoolbook_block(n, m, p, 0, j_off, 4, ni=2, nj=2, nk=1):
            all_U.append(u)
            all_V.append(v)
            all_W.append(w)
    
    rank = len(all_U)
    U_mat = np.array(all_U).T  # shape (n*m, rank)
    V_mat = np.array(all_V).T  # shape (m*p, rank)
    W_mat = np.array(all_W).T  # shape (p*n, rank)
    
    # Verify
    max_err = _verify_decomp(U_mat, V_mat, W_mat)
    if max_err > 1e-10:
        print(f"Block decomposition verification failed with max_err={max_err}")
        return None
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # First, try the block decomposition with Strassen
    try:
        result = _build_block_decomposition()
        if result is not None:
            return result
    except Exception as e:
        print(f"Block decomposition failed: {e}")
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
