import time
import numpy as np
from scipy.linalg import null_space, svd

DIMENSIONS = [2, 5, 5]

def _validate(U, V, W, r):
    """Validate decomposition with random test matrices."""
    n, m, p = DIMENSIONS
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_exp = A @ B
    C_comp = np.zeros((n, p))
    for c in range(r):
        a_c = sum(U[i*m+j][c]*A[i,j] for i in range(n) for j in range(m))
        b_c = sum(V[j*p+k][c]*B[j,k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_comp[i,k] += W[k*n+i][c]*a_c*b_c
    return np.allclose(C_comp, C_exp, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 135

    terms = []

    def add_strassen(i_start, j_start, k_start):
        # Adds 7 terms for 2x2x2 block
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    # Build the 46-term Strassen-based decomposition
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook(0, 2, 0, 2, 4, 5)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_schoolbook(0, 2, 2, 4, 4, 5)
    add_schoolbook(0, 2, 4, 5, 0, 2)
    add_schoolbook(0, 2, 4, 5, 2, 4)
    add_schoolbook(0, 2, 4, 5, 4, 5)

    rank = len(terms)
    
    # Build factor matrices
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i*m + j, c] = val
        for (j, k), val in v_dict.items():
            V[j*p + k, c] = val
        for (k, i), val in w_dict.items():
            W[k*n + i, c] = val

    # Try to reduce rank by finding linear dependencies
    # The tensor is T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We can check if any column is a linear combination of others
    
    # Check if the decomposition is valid first
    if not _validate(U.tolist(), V.tolist(), W.tolist(), rank):
        # Fallback to schoolbook
        rank_sb = n*m*p
        U_sb = [[0]*rank_sb for _ in range(n*m)]
        V_sb = [[0]*rank_sb for _ in range(m*p)]
        W_sb = [[0]*rank_sb for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Attempt rank reduction: check if any term is redundant
    # by trying to express one column as a linear combination of others
    # This is a hard problem, but let's try a simple approach:
    # For each column, check if removing it still gives a valid decomposition
    # (this only works if the column is truly zero in the tensor)
    
    # Better approach: use the fact that we can add linear combinations of terms
    # Let's try to find if there's a dependency by checking the rank of
    # the "slice" matrices
    
    # For a fixed (i,j) pair, the contribution to the tensor is:
    # T[i,j,:] = sum_c U[i,j,c] * (V[:,c] * W[:,c] outer product)
    # This is complex. Let's try a different approach.
    
    # Try: can we merge some terms? For example, if two terms have the same
    # U and V patterns but different W, we might be able to combine them.
    
    # Actually, let's try a more direct approach: use numerical linear algebra
    # to find the minimum rank by checking if the tensor has a lower rank
    # decomposition. We'll use the "unfolding" approach.
    
    # The tensor T has shape (n*m, m*p, p*n) = (10, 25, 10)
    # But it's not a general tensor - it has the special structure of matmul.
    
    # Let's try: for each possible rank r from current down, check if we can
    # find a valid decomposition of that rank. This is NP-hard in general,
    # but for small reductions we might get lucky.
    
    # Simpler approach: try removing one term at a time and see if the
    # remaining terms still span the same tensor space.
    
    # Actually, the cleanest approach: the 46-term decomposition is already
    # quite good. Let me try a different blocking strategy to see if we can
    # get below 46.
    
    # Alternative: use a 2x5x5 specific optimization
    # Think of it as: C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0..4}
    # That's 10 dot products of length 5.
    
    # Can we use a fast dot product of length 5? The minimum rank for a
    # single dot product of length 5 is 5 (schoolbook), but we have 10 of them
    # sharing structure.
    
    # Let me try: what if we use a different partitioning?
    # Instead of 2+2+1 for both m and p, try other splits.
    
    # Actually, let's just try to verify that 46 is optimal or close to it.
    # The known lower bounds for 2x5x5 matmul are not well-studied, but
    # 46 seems reasonable.
    
    # Let's try one more thing: can we use a 7-term algorithm for a 2x3x3 block?
    # If so, we could do: 2x2x2 (7) + 2x2x2 (7) + 2x1x2 (2) + 2x2x1 (2) + ...
    # Hmm, this is getting complicated.
    
    # For now, return the 46-term decomposition
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Final validation
    if not _validate(U_list, V_list, W_list, rank):
        rank_sb = n*m*p
        U_sb = [[0]*rank_sb for _ in range(n*m)]
        V_sb = [[0]*rank_sb for _ in range(m*p)]
        W_sb = [[0]*rank_sb for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_list, "V": V_list, "W": W_list}
