import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2():
        """Return (U, V, W) for 2x2*2x2 with rank 7.
        U: (4,7), V: (4,7), W: (4,7)
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # P1 = (a00+a11)(b00+b11)
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[1][0]=1; V[2][0]=1; V[3][0]=1
        W[0][0]=1; W[1][0]=1; W[2][0]=1; W[3][0]=1
        # P2 = (a10+a11)b00
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[2][1]=1; W[3][1]=1
        # P3 = a00(b01-b11)
        U[0][2]=1
        V[1][2]=1; V[3][2]=-1
        W[1][2]=1; W[3][2]=1
        # P4 = a11(b10-b00)
        U[3][3]=1
        V[2][3]=1; V[0][3]=-1
        W[0][3]=1; W[2][3]=1
        # P5 = (a00+a01)b11
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[1][4]=1
        # P6 = (a10-a00)(b00+b01)
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        # P7 = (a01-a11)(b10+b11)
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1
        return U, V, W

    def build_4x4x4():
        """Build 4x4*4x4 using 2x2 Strassen at block level.
        A: 4x4, B: 4x4, C: 4x4
        Split into 2x2 blocks of size 2x2.
        U: (16, rank), V: (16, rank), W: (16, rank)
        Indexing: U[i*4+j], V[j*4+k], W[k*4+i]
        """
        U_s, V_s, W_s = strassen_2x2()
        # For each Strassen product s (7 total), we need to compute
        # A_s * B_s where A_s and B_s are 2x2 block matrices
        # Each entry is a 2x2 matrix, so we need rank 4 per entry (schoolbook)
        # Total rank = 7 * 4 = 28... wait, that's not right.
        # Actually: the 2x2 block Strassen gives 7 products of 2x2 matrices.
        # Each 2x2*2x2 needs rank 4 (schoolbook) or rank 7 (Strassen).
        # Using schoolbook for inner: 7*4 = 28
        # Using Strassen for inner: 7*7 = 49
        # Let's use schoolbook for inner (simpler, lower rank): 28
        
        # Wait, I need to think about this more carefully.
        # The 2x2 Strassen computes C = A*B where A,B are 2x2 block matrices
        # with 2x2 entries. It does 7 multiplications of 2x2 matrices.
        # Each 2x2*2x2 multiplication has rank 4 (schoolbook).
        # So total rank = 7 * 4 = 28.
        
        # But actually, I can also use Strassen for the inner 2x2*2x2:
        # 7 * 7 = 49. Schoolbook inner is better: 28.
        
        # Hmm, but the standard 4x4 Strassen uses 7*7=49.
        # Let me use schoolbook for inner: rank 28.
        
        inner_rank = 4  # schoolbook 2x2*2x2
        total_rank = 7 * inner_rank
        
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(16)]
        W = [[0]*total_rank for _ in range(16)]
        
        # Strassen block multipliers
        # A_s = sum over (bi,bj) of A_mults[s][bi][bj] * A[bi][bj]
        # where A[bi][bj] is the 2x2 block at position (bi,bj)
        A_block_mults = [
            [[1,0],[0,1]],   # P1: A00+A11
            [[0,0],[1,1]],   # P2: A10+A11
            [[1,0],[0,0]],   # P3: A00
            [[0,0],[0,1]],   # P4: A11
            [[1,1],[0,0]],   # P5: A00+A01
            [[-1,0],[1,0]],  # P6: A10-A00
            [[0,1],[0,-1]],  # P7: A01-A11
        ]
        B_block_mults = [
            [[1,0],[0,1]],   # P1: B00+B11
            [[1,0],[0,0]],   # P2: B00
            [[0,1],[0,-1]],  # P3: B01-B11
            [[-1,0],[1,0]],  # P4: B10-B00
            [[0,0],[0,1]],   # P5: B11
            [[1,1],[0,0]],   # P6: B00+B01
            [[0,0],[1,1]],   # P7: B10+B11
        ]
        C_block_mults = [
            [[1,0],[0,1]],   # P1: C00+C11
            [[0,0],[1,-1]],  # P2: C10-C00
            [[0,1],[0,1]],   # P3: C01+C11
            [[1,0],[1,0]],   # P4: C00+C10
            [[-1,1],[0,0]],  # P5: -C00+C01
            [[0,0],[0,1]],   # P6: C11
            [[1,0],[0,0]],   # P7: C00
        ]
        
        for s in range(7):
            for t in range(inner_rank):
                col = s * inner_rank + t
                
                # A part: for each block (bi,bj), multiply by A_block_mults[s][bi][bj]
                for bi in range(2):
                    for bj in range(2):
                        a_mult = A_block_mults[s][bi][bj]
                        if a_mult == 0: continue
                        # Inner schoolbook: U[bi*2+ii, bj*2+jj] gets 1 for each (ii,jj)
                        for ii in range(2):
                            for jj in range(2):
                                U[(bi*2+ii)*4 + (bj*2+jj)][col] += a_mult * 1
                                # V and W for inner schoolbook
                # Actually, let me do this more carefully.
                # For inner schoolbook 2x2*2x2, we have 4 terms:
                # (ii,jj) pairs: (0,0), (0,1), (1,0), (1,1)
                # U[bi*2+ii, bj*2+jj] = 1
                # V[bj*2+jj, bk*2+kk] = 1
                # W[bk*2+kk, bi*2+ii] = 1
                # Wait, I'm confusing myself. Let me redo.
                
                # For the inner 2x2*2x2 at position (s,t) where t is the inner term:
                # The inner term t corresponds to (jj, kk) in the 2x2 block.
                # U: row = (bi*2+ii, bj*2+jj), but for schoolbook, jj and kk are fixed by t
                # Actually for schoolbook 2x2*2x2:
                # term (jj,kk): U[bi*2+ii, bj*2+jj]=1 for all ii
                #                V[bj*2+jj, bk*2+kk]=1 for all bk
                #                W[bk*2+kk, bi*2+ii]=1 for all ii
                # Hmm, this isn't quite right either.
                
                # Let me think again. The inner multiplication is:
                # (A_s)_{ii,jj} * (B_s)_{jj,kk} summed over jj
                # where A_s and B_s are 2x2 matrices.
                # For schoolbook: each (jj,kk) pair is one term.
                # U[(bi*2+ii), (bj*2+jj)] = 1  -- but this depends on ii which varies
                # Actually no. In the tensor decomposition:
                # U[i*m+j] * V[j*p+k] * W[k*n+i]
                # For the inner 2x2*2x2 at block position (bi,bj) for A, (bj,bk) for B:
                # i = bi*2+ii, j = bj*2+jj, k = bk*2+kk
                # U[i*4+j] = U[(bi*2+ii)*4 + (bj*2+jj)]
                # V[j*4+k] = V[(bj*2+jj)*4 + (bk*2+kk)]
                # W[k*4+i] = W[(bk*2+kk)*4 + (bi*2+ii)]
                # For schoolbook inner: term t = jj*2+kk
                # U[(bi*2+ii)*4 + (bj*2+jj)] = 1 for all ii
                # V[(bj*2+jj)*4 + (bk*2+kk)] = 1 for all bk
                # W[(bk*2+kk)*4 + (bi*2+ii)] = 1 for all ii
                # But we need to multiply by the block multipliers!
                
                # Let me restart the inner construction properly.
                pass
        
        # OK this is getting complicated. Let me use a cleaner approach.
        # I'll build it by directly specifying each rank-1 term.
        
        # Actually, let me just use the known 4x4 Strassen decomposition.
        # The standard way: 7 Strassen products, each of which is a 2x2*2x2.
        # For each Strassen product, use schoolbook (4 terms).
        # Total: 7*4 = 28 terms.
        
        U = [[0]*28 for _ in range(16)]
        V = [[0]*28 for _ in range(16)]
        W = [[0]*28 for _ in range(16)]
        
        A_bm = A_block_mults
        B_bm = B_block_mults
        C_bm = C_block_mults
        
        for s in range(7):
            for jj in range(2):
                for kk in range(2):
                    col = s*4 + jj*2 + kk
                    # For this (s, jj, kk) term:
                    # A contribution: sum over bi,bj,ii of A_bm[s][bi][bj] * delta(bj==jj) at position (bi*2+ii, jj)
                    # Wait, I need to be more careful.
                    
                    # The term is: for fixed s, jj, kk:
                    # U[i*4+j] where i=bi*2+ii, j=bj*2+jj: gets A_bm[s][bi][bj]
                    # V[j*4+k] where j=bj*2+jj, k=bk*2+kk: gets B_bm[s][bj][bk]
                    # W[k*4+i] where k=bk*2+kk, i=ci*2+ii: gets C_bm[s][ci][cj]
                    # But we need consistency: the same jj appears in U and V, same kk in V and W.
                    
                    for bi in range(2):
                        for bj in range(2):
                            a_val = A_bm[s][bi][bj]
                            if a_val == 0: continue
                            for ii in range(2):
                                i = bi*2 + ii
                                j = bj*2 + jj
                                U[i*4+j][col] += a_val
                    for bj in range(2):
                        for bk in range(2):
                            b_val = B_bm[s][bj][bk]
                            if b_val == 0: continue
                            for jj2 in range(2):
                                for kk2 in range(2):
                                    # Only when jj2==jj and kk2==kk
                                    if jj2 != jj or kk2 != kk: continue
                                    j = bj*2 + jj
                                    k = bk*2 + kk
                                    V[j*4+k][col] += b_val
                    # Wait, this isn't right. The V part should be:
                    # V[j*4+k] where j=bj*2+jj, k=bk*2+kk
                    # For fixed jj,kk: sum over bj,bk of B_bm[s][bj][bk]
                    # V[(bj*2+jj)*4 + (bk*2+kk)] = B_bm[s][bj][bk]
                    # But this is the same for all ii! That's the issue.
                    
                    # I think I'm overcomplicating this. Let me use a different approach.
                    pass
        
        # Let me use a completely different, cleaner approach.
        # Build the 4x4x4 using the known rank-28 decomposition directly.
        # Actually, let me just verify: is rank 28 achievable for 4x4*4x4?
        # Strassen gives 7*7=49. Schoolbook gives 64.
        # The best known for 4x4*4x4 is 49 (Strassen).
        # There's no known rank < 49 for 4x4*4x4 over general fields.
        # So I should use rank 49 for the 4x4*4x4 part.
        
        # Let me rebuild with rank 49 (Strassen for both levels).
        U_s, V_s, W_s = strassen_2x2()
        total_rank = 49
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(16)]
        W = [[0]*total_rank for _ in range(16)]
        
        A_bm = A_block_mults
        B_bm = B_block_mults
        C_bm = C_block_mults
        
        for s in range(7):
            for t in range(7):
                col = s*7 + t
                # U part
                for bi in range(2):
                    for bj in range(2):
                        a_mult = A_bm[s][bi][bj]
                        if a_mult == 0: continue
                        for ii in range(2):
                            for jj in range(2):
                                u_val = U_s[ii*2+jj][t]
                                if u_val == 0: continue
                                i = bi*2 + ii
                                j = bj*2 + jj
                                U[i*4+j][col] += a_mult * u_val
                # V part
                for bj in range(2):
                    for bk in range(2):
                        b_mult = B_bm[s][bj][bk]
                        if b_mult == 0: continue
                        for jj in range(2):
                            for kk in range(2):
                                v_val = V_s[jj*2+kk][t]
                                if v_val == 0: continue
                                j = bj*2 + jj
                                k = bk*2 + kk
                                V[j*4+k][col] += b_mult * v_val
                # W part
                for ci in range(2):
                    for cj in range(2):
                        c_mult = C_bm[s][ci][cj]
                        if c_mult == 0: continue
                        for kk in range(2):
                            for ii in range(2):
                                w_val = W_s[kk*2+ii][t]
                                if w_val == 0: continue
                                k = ci*2 + kk
                                i = cj*2 + ii
                                W[k*4+i][col] += c_mult * w_val
        return U, V, W

    def build_4x4x3_schoolbook():
        """Build 4x4*4x3 using schoolbook. Rank = 4*4*3 = 48.
        U: (16, 48), V: (16, 48), W: (12, 48)
        Indexing: U[i*4+j], V[j*3+k], W[k*4+i]
        """
        rank = 48
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(12)]
        W = [[0]*rank for _ in range(12)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return U, V, W

    def build_4x4x7():
        """Combine 4x4*4x4 (Strassen, rank 49) + 4x4*4x3 (schoolbook, rank 48).
        Total rank = 97.
        U: (16, 97), V: (28, 97), W: (28, 97)
        Indexing: U[i*4+j], V[j*7+k], W[k*4+i]
        """
        U_44, V_44, W_44 = build_4x4x4()
        U_43, V_43, W_43 = build_4x4x3_schoolbook()
        
        rank_44 = 49
        rank_43 = 48
        total_rank = rank_44 + rank_43
        
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(28)]
        W = [[0]*total_rank for _ in range(28)]
        
        # Copy 4x4 part (columns 0-3 of B)
        for r in range(16):
            for c in range(rank_44):
                U[r][c] = U_44[r][c]
        for r in range(16):
            for c in range(rank_44):
                V[r][c] = V_44[r][c]
        for r in range(16):
            for c in range(rank_44):
                W[r][c] = W_44[r][c]
        
        # Add 4x3 part (columns 4-6 of B)
        col_offset = rank_44
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    kk = k + 4  # global column index in B
                    U[i*4+j][col_offset] = 1
                    V[j*7+kk][col_offset] = 1
                    W[kk*4+i][col_offset] = 1
                    col_offset += 1
        
        return {"U": U, "V": V, "W": W}

    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        result = build_4x4x7()
        
        # Verify
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        rank = U.shape[1]
        
        tensor_decomp = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_decomp[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]

        if not np.allclose(tensor_decomp, 1.0, atol=1e-9):
            # Debug: print mismatch
            diff = np.abs(tensor_decomp - 1.0)
            return build_schoolbook()

        return result
    except Exception:
        return build_schoolbook()
