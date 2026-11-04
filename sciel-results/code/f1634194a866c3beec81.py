"""Matrix multiplication 3x5x6 rank-minimal decomposition using block Strassen."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition.
    U is (4, 7), V is (4, 7), W is (4, 7) with indices:
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M1
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M2
    # Wait, let me recheck. Standard Strassen:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M1
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M2
    # Hmm, I need to be careful. Let me use the standard formulation.
    
    # Actually the standard Strassen is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M1
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M2
    #
    # Wait, that doesn't look right either. Let me just use the well-known one:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M1
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M2
    #
    # Actually I think I'm overcomplicating. Let me just construct the tensors directly.
    # The tensor T[i,j,k] = sum over c of U[i,j,c]*V[j,k,c]*W[k,i,c]
    # For schoolbook: T[i,j,k] = 1 for each (i,j,k), rank 8
    # For Strassen: rank 7
    
    # Let me define the 7 rank-1 terms explicitly.
    # Each term c contributes U_c * V_c * W_c where:
    # U_c[i,j] is a 2x2 matrix (flattened to 4 entries)
    # V_c[j,k] is a 2x2 matrix (flattened to 4 entries)  
    # W_c[k,i] is a 2x2 matrix (flattened to 4 entries)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U1[i,j]: A00+A11 means U1[0,0]=1, U1[1,1]=1, others 0
    # V1[j,k]: B00+B11 means V1[0,0]=1, V1[1,1]=1, others 0
    # W1[k,i]: contributes to C00 and C11
    # C00 gets +M1, C11 gets +M1
    # So W1[0,0]=1 (k=0,i=0 -> C00), W1[1,1]=1 (k=1,i=1 -> C11)
    
    # Term 2: M2 = (A10+A11)B00
    # U2[i,j]: A10+A11 -> U2[1,0]=1, U2[1,1]=1
    # V2[j,k]: B00 -> V2[0,0]=1
    # W2[k,i]: C10 gets +M2, so W2[0,1]=1 (k=0,i=1)
    
    # Term 3: M3 = A00(B01-B11)
    # U3[i,j]: A00 -> U3[0,0]=1
    # V3[j,k]: B01-B11 -> V3[0,1]=1, V3[1,1]=-1
    # W3[k,i]: C01 gets +M3, so W3[0,0]=1 (k=0,i=0)
    
    # Term 4: M4 = A11(B10-B00)
    # U4[i,j]: A11 -> U4[1,1]=1
    # V4[j,k]: B10-B00 -> V4[1,0]=1, V4[0,0]=-1
    # W4[k,i]: C10 gets +M4, C00 gets -M4
    # So W4[0,1]=1 (k=0,i=1 for C10), W4[0,0]=-1 (k=0,i=0 for C00)
    
    # Term 5: M5 = (A00+A01)B11
    # U5[i,j]: A00+A01 -> U5[0,0]=1, U5[0,1]=1
    # V5[j,k]: B11 -> V5[1,1]=1
    # W5[k,i]: C00 gets +M5, so W5[1,0]=1 (k=1,i=0)
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U6[i,j]: A10-A00 -> U6[1,0]=1, U6[0,0]=-1
    # V6[j,k]: B00+B01 -> V6[0,0]=1, V6[0,1]=1
    # W6[k,i]: C11 gets +M6, so W6[0,1]=1 (k=0,i=1)
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U7[i,j]: A01-A11 -> U7[0,1]=1, U7[1,1]=-1
    # V7[j,k]: B10+B11 -> V7[1,0]=1, V7[1,1]=1
    # W7[k,i]: C00 gets +M7, so W7[1,0]=1 (k=1,i=0)
    
    # Now let me build the arrays.
    # U is (4, 7): U[i*2+j, c]
    # V is (4, 7): V[j*2+k, c]
    # W is (4, 7): W[k*2+i, c]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1 (c=0)
    U[0*2+0][0] = 1  # U1[0,0]
    U[1*2+1][0] = 1  # U1[1,1]
    V[0*2+0][0] = 1  # V1[0,0]
    V[1*2+1][0] = 1  # V1[1,1]
    W[0*2+0][0] = 1  # W1[0,0] -> C00
    W[1*2+1][0] = 1  # W1[1,1] -> C11
    
    # Term 2 (c=1)
    U[1*2+0][1] = 1  # U2[1,0]
    U[1*2+1][1] = 1  # U2[1,1]
    V[0*2+0][1] = 1  # V2[0,0]
    W[0*2+1][1] = 1  # W2[0,1] -> C10
    
    # Term 3 (c=2)
    U[0*2+0][2] = 1  # U3[0,0]
    V[0*2+1][2] = 1  # V3[0,1]
    V[1*2+1][2] = -1  # V3[1,1]
    W[0*2+0][2] = 1  # W3[0,0] -> C01... wait
    
    # Hmm, I need to be more careful with the W indexing.
    # W[k*n+i, c] where n=2 for the sub-problem.
    # C[k,i] gets contribution from W[k*2+i, c].
    # C01 means k=0, i=1. So W[0*2+1, c] = W[1, c].
    
    # Let me redo this more carefully.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U[0,0]=1, U[1,1]=1 (A00+A11)
    # V[0,0]=1, V[1,1]=1 (B00+B11)
    # Contributes to C00 and C11
    # C00: k=0,i=0 -> W[0,0] += 1
    # C11: k=1,i=1 -> W[3,0] += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    # U[1,0]=1, U[1,1]=1 (A10+A11)
    # V[0,0]=1 (B00)
    # Contributes to C10: k=0,i=1 -> W[1,1] += 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # Term 3: M3 = A00(B01-B11)
    # U[0,0]=1 (A00)
    # V[0,1]=1, V[1,1]=-1 (B01-B11)
    # Contributes to C01: k=0,i=1 -> W[1,2] += 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    # U[1,1]=1 (A11)
    # V[1,0]=1, V[0,0]=-1 (B10-B00)
    # Contributes to C10: k=0,i=1 -> W[1,3] += 1
    # And C00: k=0,i=0 -> W[0,3] += -1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[0][3] = -1
    
    # Term 5: M5 = (A00+A01)B11
    # U[0,0]=1, U[0,1]=1 (A00+A01)
    # V[1,1]=1 (B11)
    # Contributes to C00: k=0,i=0 -> W[0,4] += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U[1,0]=1, U[0,0]=-1 (A10-A00)
    # V[0,0]=1, V[0,1]=1 (B00+B01)
    # Contributes to C11: k=1,i=1 -> W[3,5] += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U[0,1]=1, U[1,1]=-1 (A01-A11)
    # V[1,0]=1, V[1,1]=1 (B10+B11)
    # Contributes to C00: k=0,i=0 -> W[0,6] += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to build a rank-reduced decomposition using block Strassen.
    # Strategy: Partition the 3x5x6 problem into blocks and use Strassen where possible.
    # 
    # One approach: Use the fact that 3x5x6 can be decomposed using a combination of
    # smaller Strassen blocks and schoolbook terms.
    #
    # Let's try a different approach: start with schoolbook and try to find linear
    # dependencies among the 90 basis tensors to reduce rank.
    #
    # Actually, let me try a more direct approach. The known lower bound for 3x5x6
    # is around 76-80. Let me try to construct a decomposition with rank < 90.
    #
    # Simple improvement: Use Strassen on 2x2x2 sub-blocks.
    # Partition: 
    #   n=3 = 2+1
    #   m=5 = 2+2+1
    #   p=6 = 2+2+2
    #
    # This is complex. Let me try a simpler approach first.
    #
    # Alternative: Use a known rank-7 decomposition for 2x2x2 and combine with
    # schoolbook for the remaining parts.
    #
    # Let me try: partition m=5 as 2+3, and use Strassen for the 2x2x2 part.
    # Actually, let me just try to find any improvement over 90.
    
    # For now, let's use the schoolbook decomposition as a baseline and try to
    # improve it if time permits.
    
    # Schoolbook rank = 90
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Verify schoolbook
    assert _verify_decomposition(n, m, p, U_sb, V_sb, W_sb), "Schoolbook verification failed"
    
    # Try block Strassen approach:
    # Partition: n=3 as [2,1], m=5 as [2,2,1], p=6 as [2,2,2]
    # For the 2x2x2 sub-blocks, use Strassen (rank 7 instead of 8).
    # Number of 2x2x2 blocks: we can fit one 2x2x2 block using i in [0,1], j in [0,1], k in [0,1]
    # That saves 1 rank (8->7).
    #
    # More aggressively: try to cover more of the tensor with Strassen blocks.
    # 
    # Let me try a different partition: n=[2,1], m=[2,2,1], p=[2,2,2]
    # 2x2x2 blocks: 
    #   i in [0,1], j in [0,1], k in [0,1] -> one 2x2x2 block
    #   i in [0,1], j in [0,1], k in [2,3] -> another 2x2x2 block
    #   i in [0,1], j in [0,1], k in [4,5] -> another 2x2x2 block
    #   i in [0,1], j in [2,3], k in [0,1] -> another 2x2x2 block
    #   i in [0,1], j in [2,3], k in [2,3] -> another 2x2x2 block
    #   i in [0,1], j in [2,3], k in [4,5] -> another 2x2x2 block
    #   i in [0,1], j in [4,4], k in [0,1] -> 2x1x2, not 2x2x2
    #   etc.
    #
    # So we can have 6 Strassen blocks of 2x2x2, saving 6*1 = 6 rank.
    # Remaining: schoolbook for the rest.
    #
    # Total rank = 90 - 6 = 84.
    
    # Let me implement this.
    # Partition indices:
    # n_blocks: [0,1] (size 2), [2] (size 1)
    # m_blocks: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # p_blocks: [0,1] (size 2), [2,3] (size 2), [4,5] (size 2)
    
    n_parts = [(0, 2), (2, 3)]  # (start, end) exclusive
    m_parts = [(0, 2), (2, 4), (4, 5)]
    p_parts = [(0, 2), (2, 4), (4, 6)]
    
    # For each combination of (n_block, m_block, p_block):
    # If all have size 2, use Strassen (rank 7)
    # Otherwise, use schoolbook (rank = size_n * size_m * size_p)
    
    U = []
    V = []
    W = []
    
    for n_start, n_end in n_parts:
        n_size = n_end - n_start
        for m_start, m_end in m_parts:
            m_size = m_end - m_start
            for p_start, p_end in p_parts:
                p_size = p_end - p_start
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    # Use Strassen
                    sU, sV, sW = _strassen_2x2x2()
                    for i_local in range(2):
                        for c in range(7):
                            i_global = n_start + i_local
                            U.append([U_sb[i_global * m + j][c2] if False else 0 for c2 in range(1)])
                    # This is getting complicated. Let me build it differently.
                    pass
                else:
                    pass
    
    # This approach is getting messy. Let me just use the schoolbook for now
    # and return it. The Strassen improvement would save only 6 rank (90->84).
    
    # Actually, let me try to implement the block approach properly.
    # I'll build the decomposition by concatenating blocks.
    
    U = []
    V = []
    W = []
    
    for n_start, n_end in n_parts:
        n_size = n_end - n_start
        for m_start, m_end in m_parts:
            m_size = m_end - m_start
            for p_start, p_end in p_parts:
                p_size = p_end - p_start
                
                block_rank = n_size * m_size * p_size
                if n_size == 2 and m_size == 2 and p_size == 2:
                    # Use Strassen: rank 7 instead of 8
                    sU, sV, sW = _strassen_2x2x2()
                    # sU is (4,7), sV is (4,7), sW is (4,7)
                    # For this block, we need to map local indices to global.
                    # U rows: i_local*m_size + j_local -> global (n_start+i_local)*m + (m_start+j_local)
                    # V rows: j_local*p_size + k_local -> global (m_start+j_local)*p + (p_start+k_local)
                    # W rows: k_local*n_size + i_local -> global (p_start+k_local)*n + (n_start+i_local)
                    
                    for c in range(7):
                        # U column c
                        u_col = [0] * (n * m)
                        for i_local in range(2):
                            for j_local in range(2):
                                val = sU[i_local * 2 + j_local][c]
                                if val != 0:
                                    u_col[(n_start + i_local) * m + (m_start + j_local)] = val
                        U.append(u_col)
                        
                        v_col = [0] * (m * p)
                        for j_local in range(2):
                            for k_local in range(2):
                                val = sV[j_local * 2 + k_local][c]
                                if val != 0:
                                    v_col[(m_start + j_local) * p + (p_start + k_local)] = val
                        V.append(v_col)
                        
                        w_col = [0] * (p * n)
                        for k_local in range(2):
                            for i_local in range(2):
                                val = sW[k_local * 2 + i_local][c]
                                if val != 0:
                                    w_col[(p_start + k_local) * n + (n_start + i_local)] = val
                        W.append(w_col)
                else:
                    # Schoolbook for this block
                    for i_local in range(n_size):
                        for j_local in range(m_size):
                            for k_local in range(p_size):
                                u_col = [0] * (n * m)
                                u_col[(n_start + i_local) * m + (m_start + j_local)] = 1
                                U.append(u_col)
                                
                                v_col = [0] * (m * p)
                                v_col[(m_start + j_local) * p + (p_start + k_local)] = 1
                                V.append(v_col)
                                
                                w_col = [0] * (p * n)
                                w_col[(p_start + k_local) * n + (n_start + i_local)] = 1
                                W.append(w_col)
    
    # Transpose: U should be (n*m, rank), not (rank, n*m)
    # Currently U is a list of columns (each of length n*m)
    rank = len(U)
    U_t = [[0] * rank for _ in range(n * m)]
    V_t = [[0] * rank for _ in range(m * p)]
    W_t = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U_t[i][c] = U[c][i]
        for j in range(m * p):
            V_t[j][c] = V[c][j]
        for k in range(p * n):
            W_t[k][c] = W[c][k]
    
    # Verify
    if not _verify_decomposition(n, m, p, U_t, V_t, W_t):
        # Fall back to schoolbook
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    elapsed = time.time() - start_time
    if elapsed > 100:  # Safety margin
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_t, "V": V_t, "W": W_t}
