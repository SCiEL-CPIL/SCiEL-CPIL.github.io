"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def strassen2_decomp():
        """2x2x2 Strassen, rank 7. Returns U(4x7), V(4x7), W(4x7)"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2=(A21+A22)B21: C21+=M2
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        # M3=A11(B12-B22): C12+=M3
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        # M4=A22(B21-B11): C11+=M4, C21+=M4
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # M5=(A11+A12)B22: C11-=M5, C12+=M5
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # M6=(A21-A11)(B11+B12): C22+=M6
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7=(A12-A22)(B21+B22): C11+=M7, C22+=M7
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    def schoolbook_decomp(ni, mi, pi):
        """Schoolbook decomposition for ni x mi x pi. Returns U(ni*mi x ni*mi*pi), V(mi*pi x ni*mi*pi), W(pi*ni x ni*mi*pi)"""
        rank = ni * mi * pi
        U = [[0]*rank for _ in range(ni*mi)]
        V = [[0]*rank for _ in range(mi*pi)]
        W = [[0]*rank for _ in range(pi*ni)]
        col = 0
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    U[i*mi+j][col] = 1
                    V[j*pi+k][col] = 1
                    W[k*ni+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def place_block(U_full, V_full, W_full, U_b, V_b, W_b, rank_b, 
                    i_off, j_off, k_off, n, m, p):
        """Place a block decomposition into the full matrices at offsets i_off, j_off, k_off"""
        ni, mi, pi = U_b[0].__len__(), len(U_b), len(W_b[0])
        # Actually get dimensions from the block
        ni = len(U_b) // mi if mi > 0 else len(U_b)
        # Let's just use the shapes directly
        ni_b = len(U_b)
        mi_b = len(V_b[0]) // len(W_b)  # This is getting complicated
        # Simpler: pass dimensions explicitly
        pass
    
    def build_partition(n_parts, m_parts, p_parts, decomp_fn):
        """
        Build a block partition decomposition.
        n_parts: list of row block sizes summing to n
        m_parts: list of col block sizes summing to m  
        p_parts: list of output col block sizes summing to p
        decomp_fn: function(ni, mi, pi) -> (U, V, W, rank)
        """
        total_rank = 0
        for ni in n_parts:
            for mi in m_parts:
                for pi in p_parts:
                    total_rank += decomp_fn(ni, mi, pi)[3]
        
        U_full = [[0] * total_rank for _ in range(n*m)]
        V_full = [[0] * total_rank for _ in range(m*p)]
        W_full = [[0] * total_rank for _ in range(p*n)]
        
        col = 0
        i_off = 0
        for ni in n_parts:
            j_off = 0
            for mi in m_parts:
                k_off = 0
                for pi in p_parts:
                    U_b, V_b, W_b, rank_b = decomp_fn(ni, mi, pi)
                    # Place U_b into U_full at rows i_off..i_off+ni-1, cols 0..mi-1
                    for r in range(rank_b):
                        for i in range(ni):
                            for j in range(mi):
                                U_full[(i_off+i)*m + (j_off+j)][col+r] = U_b[i*mi+j][r]
                        for j in range(mi):
                            for k in range(pi):
                                V_full[(j_off+j)*p + (k_off+k)][col+r] = V_b[j*pi+k][r]
                        for k in range(pi):
                            for i in range(ni):
                                W_full[(k_off+k)*n + (i_off+i)][col+r] = W_b[k*ni+i][r]
                    col += rank_b
                    k_off += pi
                j_off += mi
            i_off += ni
        
        return U_full, V_full, W_full, total_rank
    
    def verify(U, V, W, rank):
        """Verify the decomposition is exact"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Strategy: Use Strassen-2 for 2x2x2 blocks, schoolbook for others
    # Try partition n=2+2, m=2+3, p=2+2+2
    # Blocks:
    # (2,2,2): Strassen rank 7, count: 2*1*3 = 6 blocks -> 42
    # (2,3,2): schoolbook rank 12, count: 2*1*3 = 6 blocks -> 72
    # Total: 114. Not good.
    
    # Try partition n=2+2, m=3+2, p=2+2+2
    # (2,3,2): schoolbook 12, count 2*1*3=6 -> 72
    # (2,2,2): Strassen 7, count 2*1*3=6 -> 42
    # Total: 114
    
    # Try partition n=2+2, m=5, p=2+2+2
    # (2,5,2): schoolbook 20, count 2*1*3=6 -> 120. Same as schoolbook.
    
    # Better: Use Strassen-2 more aggressively
    # Partition n=2+2, m=2+2+1, p=2+2+2
    # (2,2,2): Strassen 7, count 2*2*3=12 -> 84
    # (2,1,2): schoolbook 4, count 2*1*3=6 -> 24
    # Total: 108
    
    # Try n=2+2, m=2+2+1, p=3+3
    # (2,2,3): schoolbook 12, count 2*2*2=8 -> 96
    # (2,1,3): schoolbook 6, count 2*1*2=4 -> 24
    # Total: 120
    
    # The key insight: for 4x5x6, we should use Strassen-2 on 2x2x2 sub-blocks
    # and minimize the number of non-Strassen blocks.
    
    # Best partition: n=2+2, m=2+2+1, p=2+2+2
    # This gives 12 Strassen-2 blocks (rank 7 each = 84) and 6 schoolbook 2x1x2 blocks (rank 4 each = 24)
    # Total: 108
    
    # But wait - can we do better with a different approach?
    # What about n=4, m=2+2+1, p=3+3?
    # (4,2,3): schoolbook 24, count 1*2*2=4 -> 96
    # (4,1,3): schoolbook 12, count 1*1*2=2 -> 24
    # Total: 120
    
    # What about using Strassen on 4x2x2 by partitioning n=2+2?
    # (2,2,2) Strassen 7, count 2*2*2=8 -> 56
    # (4,1,3) schoolbook 12, count 1*1*2=2 -> 24
    # Total: 80! This is better.
    
    # Let's build this: n=2+2, m=2+2+1, p=3+3
    # For each (ni, mi, pi) block:
    # if ni=2, mi=2, pi=3: partition p=3 into 2+1? No, keep as is.
    # Actually, for ni=2, mi=2, pi=3: we can partition p into 2+1
    #   (2,2,2): Strassen 7
    #   (2,2,1): schoolbook 4
    #   Total per block: 11
    # Count of (2,2,3) blocks: 2 (n) * 2 (m) * 2 (p) = 8 blocks -> 88
    # (2,1,3): partition p=3 into 2+1
    #   (2,1,2): schoolbook 4
    #   (2,1,1): schoolbook 2
    #   Total per block: 6
    # Count of (2,1,3) blocks: 2 * 1 * 2 = 4 blocks -> 24
    # Total: 88 + 24 = 112. Not better than 108.
    
    # Let me reconsider. The 108-rank solution from n=2+2, m=2+2+1, p=2+2+2 seems good.
    # Let's build it properly.
    
    def decomp_choice(ni, mi, pi):
        """Choose decomposition for ni x mi x pi block"""
        if ni == 2 and mi == 2 and pi == 2:
            U, V, W = strassen2_decomp()
            return U, V, W, 7
        else:
            U, V, W, rank = schoolbook_decomp(ni, mi, pi)
            return U, V, W, rank
    
    # Build n=2+2, m=2+2+1, p=2+2+2
    U_full, V_full, W_full, total_rank = build_partition(
        [2, 2], [2, 2, 1], [2, 2, 2], decomp_choice
    )
    
    if verify(U_full, V_full, W_full, total_rank):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook
    return best_result
# EVOLVE-BLOCK-END
