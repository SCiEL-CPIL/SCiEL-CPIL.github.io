"""Matrix multiplication 4x5 by 5x6 — Hybrid Strassen/schoolbook with column elimination."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 2x2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one global decomposition.
    Each decomposition is a tuple: (U_sub, V_sub, W_sub, i_start, j_start, k_start, i_size, j_size, k_size)
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        
        # Map A indices: i in [0, i_sz), j in [0, j_sz)
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        
        # Map B indices: j in [0, j_sz), k in [0, k_sz)
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        
        # Map C indices: k in [0, k_sz), i in [0, i_sz)
        # Convention: W[k*n + i]
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W

def _build_4x4x2_strassen():
    """
    Build a rank-28 decomposition for 4x4 * 4x2 -> 4x2.
    Uses 4 Strassen 2x2x2 blocks.
    Sub-block dimensions: n_sub=4, m_sub=4, p_sub=2.
    U shape: (4*4, 28) = (16, 28)
    V shape: (4*2, 28) = (8, 28)
    W shape: (2*4, 28) = (8, 28)
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 28
    n_sub, m_sub, p_sub = 4, 4, 2
    U = [[0]*rank for _ in range(n_sub * m_sub)]
    V = [[0]*rank for _ in range(m_sub * p_sub)]
    W = [[0]*rank for _ in range(p_sub * n_sub)]
    
    for ai in range(2):
        for bj in range(2):
            block_offset = (ai * 2 + bj) * 7
            for t in range(7):
                col = block_offset + t
                # U: A[i,j] where i in 2*ai..2*ai+1, j in 2*bj..2*bj+1
                for i in range(2):
                    for j in range(2):
                        val = U2[i*2+j][t]
                        if val != 0:
                            u_row = (2*ai + i) * m_sub + (2*bj + j)
                            U[u_row][col] += val
                
                # V: B[j,k] where j in 2*bj..2*bj+1, k in 0..1
                for j in range(2):
                    for k in range(2):
                        val = V2[j*2+k][t]
                        if val != 0:
                            v_row = (2*bj + j) * p_sub + k
                            V[v_row][col] += val
                
                # W: C[i,k] where i in 2*ai..2*ai+1, k in 0..1
                for k in range(2):
                    for i in range(2):
                        val = W2[k*2+i][t]
                        if val != 0:
                            w_row = k * n_sub + (2*ai + i)
                            W[w_row][col] += val
    
    return U, V, W

def _build_kronecker_strassen_4x4x4():
    """
    Build a rank-49 decomposition for 4x4 * 4x4 -> 4x4.
    Uses Kronecker product of two 2x2x2 Strassen decompositions.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            # U[i1*2+i2, j1*2+j2] = U2[i1,j1,t1] * U2[i2,j2,t2]
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1*2+j1][t1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2*2+j2][t2]
                            if c2 == 0: continue
                            u_row = (i1*2+i2)*4 + (j1*2+j2)
                            U[u_row][col] += c1 * c2
            
            # V[j1*2+j2, k1*2+k2] = V2[j1,k1,t1] * V2[j2,k2,t2]
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1*2+k1][t1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2*2+k2][t2]
                            if c2 == 0: continue
                            v_row = (j1*2+j2)*4 + (k1*2+k2)
                            V[v_row][col] += c1 * c2
            
            # W[k1*2+k2, i1*2+i2] = W2[k1,i1,t1] * W2[k2,i2,t2]
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1*2+i1][t1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2*2+i2][t2]
                            if c2 == 0: continue
                            w_row = (k1*2+k2)*4 + (i1*2+i2)
                            W[w_row][col] += c1 * c2
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = 1 if and only if the decomposition satisfies:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor)
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's verify by checking that for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # This is O(n*m*p*rank) which is 4*5*6*101 = 12120 operations. Fast enough.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _remove_linearly_dependent_columns(U, V, W, n, m, p):
    """
    Try to remove linearly dependent columns from the decomposition.
    A column r is removable if there exist coefficients c_s for s != r such that
    for all (i,j,k): U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_{s != r} c_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    This is a complex condition. A simpler approach: check if the column is a linear combination
    of other columns in a way that preserves the tensor.
    
    For now, we'll use a heuristic: try to find columns that can be eliminated by checking
    if removing them still gives a valid decomposition.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each column, check if it can be removed
    # This is expensive: O(rank^2 * n*m*p) in the worst case
    # Let's just check a few columns
    
    removed = set()
    for r in range(rank):
        if r in removed:
            continue
        # Check if column r is redundant
        # We need: for all (i,j,k), sum_{s not in removed, s != r} c_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] 
        #   = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a system of n*m*p equations in (rank - len(removed) - 1) unknowns
        # For small rank reductions, this might be solvable
        
        # Simplified check: if the column is all zeros, remove it
        if np.all(U_np[:, r] == 0) or np.all(V_np[:, r] == 0) or np.all(W_np[:, r] == 0):
            removed.add(r)
            continue
        
        # Check if the column is a scalar multiple of another column
        for s in range(rank):
            if s == r or s in removed:
                continue
            # Check if U[:,r] = alpha * U[:,s], V[:,r] = beta * V[:,s], W[:,r] = gamma * W[:,s]
            # with alpha * beta * gamma = 1
            if np.all(U_np[:, r] != 0) and np.all(V_np[:, r] != 0) and np.all(W_np[:, r] != 0):
                alpha = U_np[0, r] / U_np[0, s] if U_np[0, s] != 0 else None
                beta = V_np[0, r] / V_np[0, s] if V_np[0, s] != 0 else None
                gamma = W_np[0, r] / W_np[0, s] if W_np[0, s] != 0 else None
                if alpha is not None and beta is not None and gamma is not None:
                    if abs(alpha * beta * gamma - 1.0) < 1e-9:
                        # Check consistency
                        if (np.allclose(U_np[:, r], alpha * U_np[:, s]) and 
                            np.allclose(V_np[:, r], beta * V_np[:, s]) and 
                            np.allclose(W_np[:, r], gamma * W_np[:, s])):
                            removed.add(r)
                            break
    
    if not removed:
        return U, V, W, rank
    
    # Filter out removed columns
    keep = [r for r in range(rank) if r not in removed]
    new_rank = len(keep)
    U_new = [[U[i][r] for r in keep] for i in range(len(U))]
    V_new = [[V[i][r] for r in keep] for i in range(len(V))]
    W_new = [[W[i][r] for r in keep] for i in range(len(W))]
    
    return U_new, V_new, W_new, new_rank

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 140

    decomps = []
    
    # Block 1: 4x4x4 (i=0..3, j=0..3, k=0..3)
    U4, V4, W4 = _build_kronecker_strassen_4x4x4()
    decomps.append((U4, V4, W4, 0, 0, 0, 4, 4, 4))
    
    # Block 2: 4x4x2 (i=0..3, j=0..3, k=4..5)
    U5, V5, W5 = _build_4x4x2_strassen()
    decomps.append((U5, V5, W5, 0, 0, 4, 4, 4, 2))
    
    # Block 3: 4x1x6 (i=0..3, j=4, k=0..5)
    U6, V6, W6 = _build_schoolbook(4, 1, 6)
    decomps.append((U6, V6, W6, 0, 4, 0, 4, 1, 6))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Try to remove linearly dependent columns
    if time.time() < deadline - 20:
        U, V, W, new_rank = _remove_linearly_dependent_columns(U, V, W, n, m, p)
    
    # Verify correctness
    if time.time() < deadline - 5:
        valid = _verify_decomposition(U, V, W, n, m, p)
        if not valid:
            # Fallback: rebuild without elimination
            U, V, W = _combine_decompositions(decomps, n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
