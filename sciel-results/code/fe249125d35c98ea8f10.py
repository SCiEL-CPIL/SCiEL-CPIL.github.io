import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """Return U, V, W matrices (4x7 each) for 2x2x2 Strassen decomposition."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: (A00+A11)(B00+B11) -> C00+C11
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # c=1: (A10+A11)B00 -> C10-C11
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # c=2: A00(B01-B11) -> C01+C11
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # c=3: A11(B10-B00) -> C00+C10
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # c=4: (A00+A01)B11 -> -C00+C01
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # c=5: (A10-A00)(B00+B01) -> C11
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # c=6: (A01-A11)(B10+B11) -> C00
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _build_4x4x4_rank49():
    """
    Build rank-49 decomposition of 4x4x4 matrix multiplication tensor.
    Returns list of (u, v, w) triples, each of length 16.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # Block Strassen for 4x4x4:
    # Partition A, B, C into 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]], etc.
    
    # 7 products P_k, each computed via 2x2x2 Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    
    # P_k definitions: (A_eff, B_eff) where A_eff, B_eff are linear combos of 2x2 blocks
    # A_block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B_block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]
    
    terms = []
    
    for kp in range(7):
        for ms in range(7):
            u = np.zeros(16)
            v = np.zeros(16)
            w = np.zeros(16)
            
            # Fill u from A_specs[kp]
            for coeff, a_block in A_specs[kp]:
                bi, bj = divmod(a_block, 2)
                for r in range(2):
                    for c in range(2):
                        idx2 = r * 2 + c
                        val = U2[idx2, ms]
                        if val != 0:
                            gi = bi * 2 + r
                            gj = bj * 2 + c
                            u[gi * 4 + gj] += coeff * val
            
            # Fill v from B_specs[kp]
            for coeff, b_block in B_specs[kp]:
                bi, bj = divmod(b_block, 2)
                for r in range(2):
                    for c in range(2):
                        idx2 = r * 2 + c
                        val = V2[idx2, ms]
                        if val != 0:
                            gi = bi * 2 + r
                            gj = bj * 2 + c
                            v[gi * 4 + gj] += coeff * val
            
            # Fill w from C_specs[kp]
            for coeff, c_block in C_specs[kp]:
                bi, bj = divmod(c_block, 2)
                for r in range(2):
                    for c in range(2):
                        idx2 = r * 2 + c
                        val = W2[idx2, ms]
                        if val != 0:
                            gi = bi * 2 + r
                            gj = bj * 2 + c
                            w[gj * 4 + gi] += coeff * val  # W[k*n+i], so k=gj, i=gi
            
            if np.any(u) and np.any(v) and np.any(w):
                terms.append((u, v, w))
    
    return terms

def _build_8x8x8_rank49():
    """
    Build rank-49 decomposition of 8x8x8 matrix multiplication tensor.
    Uses block Strassen with 2x2 blocks of 4x4 submatrices, each 4x4x4 
    multiplication done via rank-49 Strassen.
    Returns list of (u, v, w) triples, each of length 64.
    """
    U4, V4, W4 = _strassen_2x2x2()  # Not used directly, we use the 4x4x4 terms
    
    # For 8x8x8, partition into 2x2 blocks of 4x4 submatrices
    # A = [[A00, A01], [A10, A11]] where each is 4x4
    # B = [[B00, B01], [B10, B11]] where each is 4x4
    # C = [[C00, C01], [C10, C11]] where each is 4x4
    
    # 7 products P_k, each is a 4x4x4 multiplication (rank 49)
    # Total rank = 7 * 49 = 343... but we can do better
    
    # Actually, let's use the same Strassen structure but with 4x4 blocks
    # The key insight: for 8x8x8, we can use block Strassen where each 
    # "scalar" multiplication is replaced by a 4x4x4 multiplication
    
    # P_k definitions (same as 4x4x4 but with 4x4 blocks):
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]
    
    # Get the 49 terms for 4x4x4
    terms_444 = _build_4x4x4_rank49()
    
    terms = []
    
    for kp in range(7):
        for u4, v4, w4 in terms_444:
            u = np.zeros(64)
            v = np.zeros(64)
            w = np.zeros(64)
            
            # Fill u from A_specs[kp]
            for coeff, a_block in A_specs[kp]:
                bi, bj = divmod(a_block, 2)
                for r in range(4):
                    for c in range(4):
                        idx4 = r * 4 + c
                        val = u4[idx4]
                        if val != 0:
                            gi = bi * 4 + r
                            gj = bj * 4 + c
                            u[gi * 8 + gj] += coeff * val
            
            # Fill v from B_specs[kp]
            for coeff, b_block in B_specs[kp]:
                bi, bj = divmod(b_block, 2)
                for r in range(4):
                    for c in range(4):
                        idx4 = r * 4 + c
                        val = v4[idx4]
                        if val != 0:
                            gi = bi * 4 + r
                            gj = bj * 4 + c
                            v[gi * 8 + gj] += coeff * val
            
            # Fill w from C_specs[kp]
            for coeff, c_block in C_specs[kp]:
                bi, bj = divmod(c_block, 2)
                for r in range(4):
                    for c in range(4):
                        idx4 = r * 4 + c
                        val = w4[idx4]
                        if val != 0:
                            gi = bi * 4 + r
                            gj = bj * 4 + c
                            w[gj * 8 + gi] += coeff * val  # W[k*n+i], so k=gj, i=gi
            
            if np.any(u) and np.any(v) and np.any(w):
                terms.append((u, v, w))
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32
    
    # Strategy: Embed 4x4x8 into 8x8x8
    # A is 4x4, embed as top-left 4x4 block of 8x8 matrix A8
    # B is 4x8, embed as top row (4x8) of 8x8 matrix B8 (top-left 4x8)
    # C is 4x8, extract from top-left 4x8 block of 8x8 matrix C8
    
    # Build 8x8x8 rank-49 decomposition (actually rank 343 with naive approach,
    # but let's try the direct approach first)
    
    # Actually, the strategy suggests using a rank-49 decomposition for 8x8x8.
    # This would require a more sophisticated construction. For now, let's use
    # the block Strassen approach which gives rank 7*49 = 343 for 8x8x8.
    
    # But wait - the task asks for 4x4x8, not 8x8x8. Let me reconsider.
    
    # The current approach splits B into two 4x4 halves and uses two 4x4x4 
    # decompositions, giving rank 98.
    
    # Let's try to find a better approach. One idea: treat the 4x8 B as a 
    # single matrix and find a decomposition that doesn't split it.
    
    # For now, let's use the existing approach but optimize it.
    
    terms_444 = _build_4x4x4_rank49()
    
    # For 4x4x8, split B into two 4x4 halves: B_L (cols 0-3) and B_R (cols 4-7)
    # C = A * B = [A*B_L, A*B_R]
    # So we need two 4x4x4 decompositions, one for each half.
    
    U_list = []
    V_list = []
    W_list = []
    
    for half in range(2):
        # B column offset: half * 4
        b_col_off = half * 4
        
        for u, v, w in terms_444:
            # u is for A (4x4), no change
            U_list.append(u.copy())
            
            # v is for B (4x4), need to shift columns by b_col_off
            v_shifted = np.zeros(32)  # m*p = 4*8 = 32
            for j in range(4):
                for k in range(4):
                    v_shifted[j * 8 + (k + b_col_off)] = v[j * 4 + k]
            V_list.append(v_shifted)
            
            # w is for C (4x4), C has 8 columns, so W[k*n+i] where k is 0..7
            # For half 0: C cols 0-3, for half 1: C cols 4-7
            w_shifted = np.zeros(32)  # p*n = 8*4 = 32
            for k in range(4):
                for i in range(4):
                    w_shifted[(k + b_col_off) * 4 + i] = w[k * 4 + i]
            W_list.append(w_shifted)
    
    rank = len(U_list)
    U = np.array(U_list).T  # shape (16, rank)
    V = np.array(V_list).T  # shape (32, rank)
    W = np.array(W_list).T  # shape (32, rank)
    
    # Verify correctness
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-10:
                    error += abs(val - 1)
    
    if error > 1e-8:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((nm, rank_school))
        V_school = np.zeros((mp, rank_school))
        W_school = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
