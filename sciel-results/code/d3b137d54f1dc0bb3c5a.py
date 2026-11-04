"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Build the 4x4x4 Winograd decomposition (rank 49)
    # Using 2x2 block Strassen applied twice
    # Then add rank-1 term for the 5th column (4x4x1)
    
    # The 2x2 Strassen decomposition:
    # For matrices A(2x2), B(2x2), C(2x2):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B00-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For 4x4x4, partition into 2x2 blocks of 2x2 matrices:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ck is 2x2
    #
    # Apply Strassen at the block level:
    # S1 = (A00+A11)*(B00+B11) -> 2x2*2x2, use Strassen (7 terms)
    # S2 = (A10+A11)*B00 -> 2x2*2x2, use Strassen (7 terms)
    # S3 = A00*(B10-B11) -> 2x2*2x2, use Strassen (7 terms)
    # S4 = A11*(B00-B10) -> 2x2*2x2, use Strassen (7 terms)
    # S5 = (A00+A01)*B11 -> 2x2*2x2, use Strassen (7 terms)
    # S6 = (A10-A00)*(B00+B01) -> 2x2*2x2, use Strassen (7 terms)
    # S7 = (A01-A11)*(B10+B11) -> 2x2*2x2, use Strassen (7 terms)
    #
    # Total: 7*7 = 49 scalar multiplications
    
    # Let's define the Strassen decomposition for 2x2x2 first
    # U, V, W for 2x2x2 with rank 7
    # U shape: (4, 7) - rows indexed by (i,j) for A[i,j]
    # V shape: (4, 7) - rows indexed by (j,k) for B[j,k]  
    # W shape: (4, 7) - rows indexed by (k,i) for C[k,i]... 
    
    # Wait, let me be more careful with indexing.
    # For 2x2x2: A is 2x2, B is 2x2, C is 2x2
    # U[i*m+j, c] where m=2, so U has 4 rows
    # V[j*p+k, c] where p=2, so V has 4 rows
    # W[k*n+i, c] where n=2, so W has 4 rows
    
    # Strassen 2x2x2 decomposition:
    # Term 1: (A00+A11)*(B00+B11)
    #   U: A00=1, A11=1, others 0 -> U[0]=1, U[3]=1 (row 0 = i0j0, row 3 = i1j1)
    #   V: B00=1, B11=1, others 0 -> V[0]=1, V[3]=1
    #   W: contributes to C00 and C11
    #     C00 += M1, C11 += M1
    #     So W[0,0]=1 (k=0,i=0), W[1,1]=1 (k=1,i=1)
    
    # Let me build this systematically
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition, rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Row indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        # i,j,k in {0,1}
        
        # Term 0: M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00 (k=0,i=0)
        W[3, 0] = 1  # C11 (k=1,i=1)
        
        # Term 1: M2 = (A10+A11)*B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10 (k=0,i=1)
        # Wait, C10 = M2 + M4, so M2 contributes to C10
        # C10 is at position k=0, i=1 -> W[0*2+1] = W[1]
        # Let me redo: W[k*n+i] where n=2
        # C10 means row 1, col 0 of C, so i=1, k=0 -> W[0*2+1] = W[1]
        W[1, 1] = 1
        
        # Term 2: M3 = A00*(B10-B11)
        U[0, 2] = 1  # A00
        V[2, 2] = 1  # B10
        V[3, 2] = -1 # B11
        # C01 = M3 + M5, so M3 contributes to C01
        # C01: i=0, k=1 -> W[1*2+0] = W[2]
        W[2, 2] = 1
        
        # Term 3: M4 = A11*(B00-B10)
        U[3, 3] = 1  # A11
        V[0, 3] = 1  # B00
        V[2, 3] = -1 # B10
        # C00 = M1 + M4 - M5 - M7, C10 = M2 + M4
        # M4 contributes to C00 and C10
        W[0, 3] = 1  # C00: k=0,i=0
        W[1, 3] = 1  # C10: k=0,i=1
        
        # Term 4: M5 = (A00+A01)*B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        # C00 = M1 + M4 - M5 - M7, C01 = M3 + M5
        W[0, 4] = -1 # C00: k=0,i=0
        W[2, 4] = 1  # C01: k=1,i=0
        
        # Term 5: M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1  # A10
        U[0, 5] = -1 # A00
        V[0, 5] = 1  # B00
        V[1, 5] = 1  # B01
        # C11 = M1 - M2 + M3 + M7... wait
        # Standard Strassen: C11 = M1 - M2 + M3 + M7
        # But M6 is not used in the standard 7-term Strassen!
        # Let me recheck. Standard Strassen uses 7 multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # M6 is not used! That's a mistake. Let me use the correct 7-term form.
        # Actually, the standard Strassen does use all 7. Let me recheck.
        # C11 = M1 - M2 + M3 + M7... but where does M6 go?
        # I think I'm confusing myself. Let me use a verified Strassen decomposition.
        
        return U, V, W
    
    # Let me use a verified approach. I'll construct the 4x4x4 decomposition
    # by directly building the Winograd algorithm.
    
    # Alternative: Use the fact that we can verify any decomposition.
    # Let me try to construct a valid rank-49 decomposition for 4x4x4.
    
    # Actually, let me just use the schoolbook for now but try to find a better one.
    # The key question is: can I find a valid low-rank decomposition within the time limit?
    
    # Let me try a different approach: use the known Winograd coefficients.
    
    # For 4x4x4, the Winograd algorithm can be described as follows:
    # Partition A, B into 2x2 blocks of 2x2 matrices.
    # Apply Strassen at the block level, where each block multiplication
    # is done via Strassen on 2x2 matrices.
    
    # Let me carefully construct the 2x2 Strassen first and verify it.
    
    def build_strassen_2x2x2():
        """Build verified 2x2x2 Strassen decomposition, rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        # i,j,k in {0,1}
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # C00, C11
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1  # C10 (k=0, i=1)
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1  # C01 (k=1, i=0)
        
        # M4 = A11(B00-B10)
        U[3, 3] = 1
        V[0, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1  # C00, C10
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1  # C00 (neg), C01
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C11 (k=1, i=1)
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[3, 6] = 1  # C00 (neg), C11
        
        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = np.dot(np.dot(U[i*2+j], V[j*2+k]), W[k*2+i])
                    if abs(val - 1.0) > 1e-10:
                        raise ValueError(f"Strassen 2x2x2 verification failed at {i},{j},{k}: {val}")
        
        return U, V, W
    
    # Now build 4x4x4 using block Strassen
    # A is 4x4, partition into 2x2 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]]
    # Where A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # Similarly for B (4x4) and C (4x4)
    
    # Block Strassen:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B10-B11)
    # S4 = A11(B00-B10)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    #
    # C00 = S1 + S4 - S5 - S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S7
    
    # Each Sj is a 2x2 matrix computed as product of two 2x2 matrices.
    # Each such product uses Strassen (rank 7).
    
    # So total rank = 7 * 7 = 49.
    
    # Let's build this.
    U444 = np.zeros((16, 49))  # 4*4 = 16 rows
    V444 = np.zeros((16, 49))  # 4*4 = 16 rows
    W444 = np.zeros((16, 49))  # 4*4 = 16 rows
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = build_strassen_2x2x2()
    
    # Block mapping:
    # A block (bi, bj) where bi in {0,1}, bj in {0,1}
    # A[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
    # Row index in U for element A[i,j]: i*4+j
    # For block (bi,bj), element A[i,j] where i=bi*2+di, j=bj*2+dj
    # U index: (bi*2+di)*4 + (bj*2+dj) = bi*8 + bj*4 + di*4 + dj
    
    # Similarly for B: B[j,k], block (bj,bk)
    # V index: j*4+k where j=bj*2+dj, k=bk*2+dk
    # V index: bj*8 + bk*4 + dj*4 + dk
    
    # For C: C[i,k], block (bi,bk)
    # W index: k*4+i where k=bk*2+dk, i=bi*2+di
    # W index: bk*8 + bi*4 + dk*4 + di
    
    # Block Strassen terms:
    # For each block-level term t in 0..6:
    #   For each inner Strassen term s in 0..6:
    #     Column c = t*7 + s
    #     U: sum over (bi,bj) in the A-blocks involved, with coefficients from block Strassen
    #        times U2[di*2+dj, s]
    #     V: sum over (bj,bk) in the B-blocks involved, with coefficients from block Strassen
    #        times V2[dj*2+dk, s]
    #     W: sum over (bi,bk) in the C-blocks involved, with coefficients from block Strassen
    #        times W2[dk*2+di, s]
    
    # Block Strassen definitions:
    # S1 = (A00+A11)(B00+B11): A blocks (0,0),(1,1); B blocks (0,0),(1,1)
    # S2 = (A10+A11)B00: A blocks (1,0),(1,1); B blocks (0,0)
    # S3 = A00(B10-B11): A blocks (0,0); B blocks (1,0),(1,1)
    # S4 = A11(B00-B10): A blocks (1,1); B blocks (0,0),(1,0)
    # S5 = (A00+A01)B11: A blocks (0,0),(0,1); B blocks (1,1)
    # S6 = (A10-A00)(B00+B01): A blocks (1,0),(0,0); B blocks (0,0),(0,1)
    # S7 = (A01-A11)(B10+B11): A blocks (0,1),(1,1); B blocks (1,0),(1,1)
    
    # C block contributions:
    # C00 = S1 + S4 - S5 - S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S7
    
    # Define block Strassen structure:
    # For each term t, list of (block_index, coefficient) for A, B, C
    
    # A blocks: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B blocks: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C blocks: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    block_strassen = {
        0: {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},  # S1
        1: {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1)]},  # S2 -> C10
        2: {'A': [(0, 1)], 'B': [(2, 1), (3, -1)], 'C': [(1, 1)]},  # S3 -> C01
        3: {'A': [(3, 1)], 'B': [(0, 1), (2, -1)], 'C': [(0, 1), (2, 1)]},  # S4 -> C00, C10
        4: {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},  # S5 -> C00(neg), C01
        5: {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},  # S6 -> C11
        6: {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, -1), (3, 1)]},  # S7 -> C00(neg), C11
    }
    
    # Now build U444, V444, W444
    for t in range(7):
        for s in range(7):
            c = t * 7 + s
            # U part
            for (ab, acoeff) in block_strassen[t]['A']:
                bi, bj = divmod(ab, 2)
                for di in range(2):
                    for dj in range(2):
                        u_idx = bi * 8 + bj * 4 + di * 4 + dj
                        U444[u_idx, c] += acoeff * U2[di * 2 + dj, s]
            
            # V part
            for (bb, bcoeff) in block_strassen[t]['B']:
                bj, bk = divmod(bb, 2)
                for dj in range(2):
                    for dk in range(2):
                        v_idx = bj * 8 + bk * 4 + dj * 4 + dk
                        V444[v_idx, c] += bcoeff * V2[dj * 2 + dk, s]
            
            # W part
            for (cb, ccoeff) in block_strassen[t]['C']:
                bi, bk = divmod(cb, 2)
                for di in range(2):
                    for dk in range(2):
                        w_idx = bk * 8 + bi * 4 + dk * 4 + di
                        W444[w_idx, c] += ccoeff * W2[dk * 2 + di, s]
    
    # Verify 4x4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.dot(np.dot(U444[i*4+j], V444[j*4+k]), W444[k*4+i])
                if abs(val - 1.0) > 1e-10:
                    # Fall back to schoolbook
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        # Verification passed, now add the 4x4x1 part (rank 1)
        # C[:,4] = A @ B[:,4]
        # This is a matrix-vector product. The tensor T[i,j] = 1 for all i,j.
        # Rank 1: U[i*4+j, 49] = 1, V[j*4+4, 49] = 1, W[4*4+i, 49] = 1
        # Sum_j A[i,j] * B[j,4] * 1 = (A @ B[:,4])[i]
        
        U_full = np.zeros((16, 50))
        V_full = np.zeros((20, 50))
        W_full = np.zeros((20, 50))
        
        U_full[:, :49] = U444
        V_full[:16, :49] = V444
        W_full[:16, :49] = W444
        
        # Rank 1 term for column 4 (index 4 in B, so j*4+4 for j in 0..3)
        # U[i*4+j, 49] = 1 for all i,j
        U_full[:, 49] = 1
        # V[j*4+4, 49] = 1 for j in 0..3
        for j in range(4):
            V_full[j*4+4, 49] = 1
        # W[4*4+i, 49] = W[16+i, 49] = 1 for i in 0..3
        for i in range(4):
            W_full[16+i, 49] = 1
        
        # Verify full 4x4x5 decomposition
        valid = True
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    val = np.dot(np.dot(U_full[i*4+j], V_full[j*4+k]), W_full[k*4+i])
                    if abs(val - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to lists
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U_school = np.zeros((n * m, rank))
    V_school = np.zeros((m * p, rank))
    W_school = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
