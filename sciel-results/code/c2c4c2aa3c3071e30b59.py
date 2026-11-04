import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 140

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            T_computed += u_mat[:, :, None] * v_mat[None, :, :] * w_mat[None, :, :]
        
        return np.max(np.abs(T_computed - 1.0)) < tol
    
    def strassen_2x2_2():
        """Strassen 2x2*2x2 decomposition, rank 7.
        U indexed by [i,j] -> i*2+j
        V indexed by [j,k] -> j*2+k  
        W indexed by [k,i] -> k*2+i
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (A00+A10)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def build_3x3_3x2_optimized():
        """Build optimized decomposition for 3x3 @ 3x2."""
        n_local, m_local, p_local = 3, 3, 2
        
        # Start with Strassen for 2x2x2 block at i=0,1 j=0,1 k=0,1
        sU, sV, sW = strassen_2x2_2()
        
        # Try to find a lower-rank completion
        # The remainder positions are:
        # i=2, j=0,1,2, k=0,1 (6 positions)
        # i=0,1, j=2, k=0,1 (4 positions)
        
        # Let's try rank 16: 7 (Strassen) + 9 (remainder)
        # Or try to find if we can do better
        
        # Approach: use the fact that for fixed i, the slice is a matrix
        # T[i,j,k] for fixed i is a 3x2 matrix
        # We need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Let's try a structured approach:
        # For i=2: T[2,j,k] = sum_r U[2,j,r] * V[j,k,r] * W[k,2,r]
        # For i=0,1: T[i,2,k] = sum_r U[i,2,r] * V[2,k,r] * W[k,i,r]
        
        # Try rank 16 construction
        rank = 16
        U = np.zeros((m_local * n_local, rank))  # (i,j) -> i*m+j
        V = np.zeros((m_local * p_local, rank))  # (j,k) -> j*p+k
        W = np.zeros((p_local * n_local, rank))  # (k,i) -> k*n+i
        
        # Place Strassen in first 7 columns
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m_local + j, c] = sU[i * 2 + j, c]
            for j in range(2):
                for k in range(2):
                    V[j * p_local + k, c] = sV[j * 2 + k, c]
            for k in range(2):
                for i in range(2):
                    W[k * n_local + i, c] = sW[k * 2 + i, c]
        
        # Now handle remainder with columns 7-15
        # We need to satisfy:
        # For i=2, j=0,1,2, k=0,1: T[2,j,k] = 1
        # For i=0,1, j=2, k=0,1: T[i,2,k] = 1
        
        # Let's use a simple but effective approach:
        # For each (i,j,k) in remainder, assign one rank term
        # But we have 10 remainder positions and only 9 columns left
        
        # Alternative: share some rank terms
        # Key insight: V[j,k,r] appears in both i=2 and i=0,1 terms
        
        # Let's try: for j=2, k=0,1 use 2 rank terms that handle both i=0 and i=1
        # U[0,2,r] and U[1,2,r] can be different, so we need separate terms
        
        # Actually, let's just use 10 rank terms for remainder (total 17)
        # But try to see if we can merge any
        
        # For now, use the straightforward approach
        col = 7
        remainder_positions = []
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    if not (i < 2 and j < 2):  # Not covered by Strassen
                        remainder_positions.append((i, j, k))
        
        for (i, j, k) in remainder_positions:
            if col >= rank:
                break
            U[i * m_local + j, col] = 1.0
            V[j * p_local + k, col] = 1.0
            W[k * n_local + i, col] = 1.0
            col += 1
        
        # If we have more positions than columns, we need a different approach
        # For now, if col > rank, fall back to schoolbook for remainder
        if col > rank:
            # Use schoolbook for all remainder
            rank = 7 + len(remainder_positions)
            U = np.zeros((m_local * n_local, rank))
            V = np.zeros((m_local * p_local, rank))
            W = np.zeros((p_local * n_local, rank))
            
            for c in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m_local + j, c] = sU[i * 2 + j, c]
                for j in range(2):
                    for k in range(2):
                        V[j * p_local + k, c] = sV[j * 2 + k, c]
                for k in range(2):
                    for i in range(2):
                        W[k * n_local + i, c] = sW[k * 2 + i, c]
            
            col = 7
            for (i, j, k) in remainder_positions:
                U[i * m_local + j, col] = 1.0
                V[j * p_local + k, col] = 1.0
                W[k * n_local + i, col] = 1.0
                col += 1
        
        return U, V, W
    
    def build_3x3_3x2_rank16():
        """Try to construct a rank-16 decomposition for 3x3 @ 3x2."""
        n_local, m_local, p_local = 3, 3, 2
        rank = 16
        
        U = np.zeros((n_local * m_local, rank))
        V = np.zeros((m_local * p_local, rank))
        W = np.zeros((p_local * n_local, rank))
        
        sU, sV, sW = strassen_2x2_2()
        
        # Strassen part: columns 0-6
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m_local + j, c] = sU[i * 2 + j, c]
            for j in range(2):
                for k in range(2):
                    V[j * p_local + k, c] = sV[j * 2 + k, c]
            for k in range(2):
                for i in range(2):
                    W[k * n_local + i, c] = sW[k * 2 + i, c]
        
        # Remainder: columns 7-15 (9 columns)
        # Positions: (2,0,0), (2,1,0), (2,2,0), (2,0,1), (2,1,1), (2,2,1),
        #            (0,2,0), (1,2,0), (0,2,1), (1,2,1)
        # That's 10 positions, but we only have 9 columns
        
        # Strategy: share rank terms where possible
        # Note that V[j,k,r] is shared across different i values
        # And W[k,i,r] is shared across different j values
        
        # Let's try to use 9 rank terms cleverly
        # For j=2: V[2,0,r] and V[2,1,r] can be set to handle multiple i's
        
        # Idea: Use 2 rank terms for j=2, k=0: one for i=0, one for i=1, and one for i=2
        # But that's 3 terms for 3 positions
        
        # Alternative: accept rank 17
        # Or try a different Strassen embedding
        
        # Let's try: what if we use a different split?
        # Instead of Strassen on i=0,1 j=0,1, use Strassen on different blocks
        
        # For now, return rank 17
        rank_actual = 17
        U = np.zeros((n_local * m_local, rank_actual))
        V = np.zeros((m_local * p_local, rank_actual))
        W = np.zeros((p_local * n_local, rank_actual))
        
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m_local + j, c] = sU[i * 2 + j, c]
            for j in range(2):
                for k in range(2):
                    V[j * p_local + k, c] = sV[j * 2 + k, c]
            for k in range(2):
                for i in range(2):
                    W[k * n_local + i, c] = sW[k * 2 + i, c]
        
        col = 7
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    if not (i < 2 and j < 2):
                        U[i * m_local + j, col] = 1.0
                        V[j * p_local + k, col] = 1.0
                        W[k * n_local + i, col] = 1.0
                        col += 1
        
        return U, V, W
    
    # Build the full 3x3x8 decomposition
    U_block, V_block, W_block = build_3x3_3x2_rank16()
    rank_block = U_block.shape[1]
    
    rank_total = 4 * rank_block
    U = np.zeros((nm, rank_total))
    V = np.zeros((mp, rank_total))
    W = np.zeros((pn, rank_total))
    
    for g in range(4):
        k0 = 2 * g
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        for i in range(n):
            for j in range(m):
                U[i*m + j, col_start:col_end] = U_block[i*m + j, :]
        
        for j in range(m):
            for k_local in range(2):
                k_global = k0 + k_local
                V[j*p + k_global, col_start:col_end] = V_block[j*2 + k_local, :]
        
        for k_local in range(2):
            k_global = k0 + k_local
            for i in range(n):
                W[k_global*n + i, col_start:col_end] = W_block[k_local*n + i, :]
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    
    # Fallback to schoolbook
    return schoolbook()

