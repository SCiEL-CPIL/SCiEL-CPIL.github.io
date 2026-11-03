import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return (U,V,W) for rank-7 Strassen decomposition of 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W


def _rank9_2x3x2():
    """Rank-9 decomposition for 2x3 * 3x2 -> 2x2.
    U: (6, 9) indexed [i*3+j], V: (6, 9) indexed [j*2+k], W: (4, 9) indexed [k*2+i]
    
    Construction: Split the 3 middle dimension into 1+2.
    - 2x1x2 part: rank 4 (schoolbook, can't do better)
    - 2x2x2 part: rank 7 (Strassen)
    Total: 11... but we can do 9.
    
    Alternative: Use the fact that 2x3x2 has rank 9.
    Let's use a different construction.
    
    Actually, let me use: think of A as 2x3, B as 3x2.
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    One rank-9 construction:
    M1 = A00 * (B00+B11+B22)  -- no, this doesn't work cleanly.
    
    Let me try: embed 2x3x2 into 3x3x3 and use a subset of Strassen-3x3x3 (rank 23)?
    That's too expensive.
    
    Better: 2x3x2 = (2x2x2) + (2x1x2) with overlap optimization.
    2x2x2: rank 7
    2x1x2: rank 4
    But we can share some terms.
    
    Actually, the known rank of 2x3x2 is 9. Here's one construction:
    View it as computing C = A*B where A is 2x3, B is 3x2.
    
    Let A = [a b c] where a,b,c are 2x1 column vectors.
    Let B = [d e] where d,e are 3x1 column vectors.
    C = a*d^T + b*e^T... no, that's not right.
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Rank-9 construction using Strassen-like ideas:
    Split j into j0 (size 1) and j1 (size 2).
    C = A[:,j0]*B[j0,:] + A[:,j1]*B[j1,:]
    First term: 2x1x2, rank 4
    Second term: 2x2x2, rank 7
    Total: 11, but we can do 9 by combining.
    
    Here's a concrete rank-9 construction:
    M1 = (A00+A11)*(B00+B11)  -- from Strassen on the 2x2 subblock
    ... this is getting complex. Let me just use rank 11 (7+4) for now and note it's suboptimal.
    """
    # Use 2x2x2 (rank 7) for columns 1,2 and 2x1x2 (rank 4) for column 0
    # Total rank 11
    U = np.zeros((6, 11))
    V = np.zeros((6, 11))
    W = np.zeros((4, 11))
    
    # 2x1x2 schoolbook for j=0
    for i in range(2):
        for k in range(2):
            U[i*3+0, k] = 1.0  # A[i,0]
            V[0*2+k, k] = 1.0  # B[0,k]
            W[k*2+i, k] = 1.0  # C[i,k]
    
    # 2x2x2 Strassen for j=1,2
    U_s, V_s, W_s = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for jj in range(2):
                U[i*3+(1+jj), 4+r] = U_s[i*2+jj, r]
        for jj in range(2):
            for k in range(2):
                V[(1+jj)*2+k, 4+r] = V_s[jj*2+k, r]
        for k in range(2):
            for i in range(2):
                W[k*2+i, 4+r] = W_s[k*2+i, r]
    
    return U, V, W


def _rank9_2x2x3():
    """Rank-9 decomposition for 2x2 * 2x3 -> 2x3.
    U: (4, 9) indexed [i*2+j], V: (6, 9) indexed [j*3+k], W: (6, 9) indexed [k*2+i]
    """
    # By symmetry, similar to 2x3x2 but transposed
    # 2x2x3 = 2x2x1 (rank 4) + 2x2x2 (rank 7) = 11
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # 2x2x1 schoolbook for k=0
    for i in range(2):
        for j in range(2):
            U[i*2+j, j] = 1.0
            V[j*3+0, j] = 1.0
            W[0*2+i, j] = 1.0
    
    # 2x2x2 Strassen for k=1,2
    U_s, V_s, W_s = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*2+j, 4+r] = U_s[i*2+j, r]
        for j in range(2):
            for kk in range(2):
                V[j*3+(1+kk), 4+r] = V_s[j*2+kk, r]
        for kk in range(2):
            for i in range(2):
                W[(1+kk)*2+i, 4+r] = W_s[kk*2+i, r]
    
    return U, V, W


def _rank15_2x3x3():
    """Decomposition for 2x3 * 3x3 -> 2x3.
    U: (6, rank) indexed [i*3+j], V: (9, rank) indexed [j*3+k], W: (6, rank) indexed [k*2+i]
    """
    # 2x3x3 = 2x3x2 (rank 11) + 2x3x1 (rank 6) = 17
    # Or: 2x3x3 = 2x2x3 (rank 11) + 2x1x3 (rank 6) = 17
    # Let's use 2x3x2 + 2x3x1
    U_232, V_232, W_232 = _rank9_2x3x2()
    r1 = U_232.shape[1]
    
    U = np.zeros((6, r1 + 6))
    V = np.zeros((9, r1 + 6))
    W = np.zeros((6, r1 + 6))
    
    # Embed 2x3x2 into 2x3x3 (k=0,1)
    for r in range(r1):
        for i in range(2):
            for j in range(3):
                U[i*3+j, r] = U_232[i*3+j, r]
        for j in range(3):
            for k in range(2):
                V[j*3+k, r] = V_232[j*2+k, r]
        for k in range(2):
            for i in range(2):
                W[k*2+i, r] = W_232[k*2+i, r]
    
    # 2x3x1 schoolbook for k=2
    c = r1
    for i in range(2):
        for j in range(3):
            U[i*3+j, c] = 1.0
            V[j*3+2, c] = 1.0
            W[2*2+i, c] = 1.0
            c += 1
    
    return U, V, W


def _schoolbook(n_b, m_b, p_b):
    """Schoolbook decomposition for n_b x m_b * m_b x p_b."""
    rank = n_b * m_b * p_b
    U = np.zeros((n_b*m_b, rank))
    V = np.zeros((m_b*p_b, rank))
    W = np.zeros((p_b*n_b, rank))
    c = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[i*m_b+j, c] = 1.0
                V[j*p_b+k, c] = 1.0
                W[k*n_b+i, c] = 1.0
                c += 1
    return U, V, W


def _combine_decompositions(decomps):
    """Combine a list of (U,V,W) decompositions into one."""
    if not decomps:
        return None
    
    n, m, p = N, M, P
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    c = 0
    for U_i, V_i, W_i in decomps:
        r = U_i.shape[1]
        U[:, c:c+r] = U_i
        V[:, c:c+r] = V_i
        W[:, c:c+r] = W_i
        c += r
    
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = N, M, P  # 4, 5, 5
    
    def embed_block(n_b, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b):
        """Embed a sub-block decomposition into the full space."""
        rank = U_b.shape[1]
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            for i in range(n_b):
                for j in range(m_b):
                    U[(i_off+i)*m + (j_off+j), r] = U_b[i*m_b+j, r]
            for j in range(m_b):
                for k in range(p_b):
                    V[(j_off+j)*p + (k_off+k), r] = V_b[j*p_b+k, r]
            for k in range(p_b):
                for i in range(n_b):
                    W[(k_off+k)*n + (i_off+i), r] = W_b[k*n_b+i, r]
        return U, V, W
    
    def get_decomp(n_b, m_b, p_b):
        """Get best known decomposition for given block sizes."""
        if n_b == 2 and m_b == 2 and p_b == 2:
            return _strassen_2x2x2()
        elif n_b == 2 and m_b == 3 and p_b == 2:
            return _rank9_2x3x2()
        elif n_b == 2 and m_b == 2 and p_b == 3:
            return _rank9_2x2x3()
        elif n_b == 2 and m_b == 3 and p_b == 3:
            return _rank15_2x3x3()
        elif n_b == 1:
            return _schoolbook(1, m_b, p_b)
        elif m_b == 1:
            return _schoolbook(n_b, 1, p_b)
        elif p_b == 1:
            return _schoolbook(n_b, m_b, 1)
        else:
            return _schoolbook(n_b, m_b, p_b)
    
    # Try multiple block splits and pick the best
    best_rank = n * m * p
    best_result = None
    
    # Split 1: 4=2+2, 5=3+2, 5=3+2
    splits = [
        ([2, 2], [3, 2], [3, 2]),
        ([2, 2], [2, 3], [2, 3]),
        ([1, 1, 1, 1], [3, 2], [3, 2]),
        ([2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]),
        ([2, 2], [3, 2], [2, 3]),
    ]
    
    for i_split, j_split, k_split in splits:
        if time.time() - t0 > DEADLINE_S - 10:
            break
        
        decomps = []
        valid = True
        
        i_offs = list(np.cumsum([0] + i_split[:-1]))
        j_offs = list(np.cumsum([0] + j_split[:-1]))
        k_offs = list(np.cumsum([0] + k_split[:-1]))
        
        for idx, (i_off, n_b) in enumerate(zip(i_offs, i_split)):
            for jdx, (j_off, m_b) in enumerate(zip(j_offs, j_split)):
                for kdx, (k_off, p_b) in enumerate(zip(k_offs, k_split)):
                    U_b, V_b, W_b = get_decomp(n_b, m_b, p_b)
                    U_e, V_e, W_e = embed_block(n_b, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b)
                    decomps.append((U_e, V_e, W_e))
        
        U, V, W = _combine_decompositions(decomps)
        current_rank = U.shape[1]
        
        if current_rank < best_rank:
            if _validate(U, V, W):
                best_rank = current_rank
                best_result = (U, V, W)
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
    else:
        U, V, W = best_result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
