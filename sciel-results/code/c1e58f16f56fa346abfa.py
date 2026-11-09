"""Program for rank-minimal tensor decomposition of matrix multiplication 3x4x7."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    max_diff = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > max_diff:
                    max_diff = abs(s - 1.0)
    return max_diff < 1e-9

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # 120s limit, keep margin

    n, m, p = DIMENSIONS

    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W, "rank": rank}

    # Strassen 2x2x2 decomposition
    def strassen_222():
        """
        Returns U, V, W for 2x2x2 Strassen.
        U: 4x7, V: 4x7, W: 4x7
        Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1  # C00 += M1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # C10 += M2 (W index k*2+i, C10 -> k=0, i=1 -> W[0*2+1]=W[2])
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1  # C01 += M3 (W index k*2+i, C01 -> k=1, i=0 -> W[1*2+0]=W[2]? No. C[i,k]. W[k*n+i]. n=2. C01 -> i=0, k=1. W[1*2+0] = W[2]. Wait.
        # Let's re-verify W indexing.
        # Task says: W[k*n+i, r].
        # C[i,k] is the element.
        # In Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # W[k*n+i, c] corresponds to coefficient of C[i,k] in the c-th term.
        # So for C00 (i=0, k=0): W[0*2+0, c] = W[0, c]
        # For C01 (i=0, k=1): W[1*2+0, c] = W[2, c]
        # For C10 (i=1, k=0): W[0*2+1, c] = W[1, c]
        # For C11 (i=1, k=1): W[1*2+1, c] = W[3, c]
        
        # Let's re-assign W correctly based on this.
        
        # M1: C00 += 1, C11 += 1
        W[0][0] = 1
        W[3][0] = 1
        
        # M2: C10 += 1
        W[1][1] = 1
        
        # M3: C01 += 1
        W[2][2] = 1
        
        # M4: C00 += 1, C10 += 1
        W[0][3] = 1
        W[1][3] = 1
        
        # M5: C01 += 1, C00 -= 1
        W[2][4] = 1
        W[0][4] = -1
        
        # M6: C11 += 1
        W[3][5] = 1
        
        # M7: C00 += 1
        W[0][6] = 1
        
        return U, V, W

    def schoolbook_block(n_b, m_b, p_b):
        """Schoolbook for n_b x m_b x p_b."""
        rank = n_b * m_b * p_b
        U = [[0]*rank for _ in range(n_b*m_b)]
        V = [[0]*rank for _ in range(m_b*p_b)]
        W = [[0]*rank for _ in range(p_b*n_b)]
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i*m_b+j][col] = 1
                    V[j*p_b+k][col] = 1
                    W[k*n_b+i][col] = 1
                    col += 1
        return U, V, W

    def build_full_decomp():
        """
        Build 3x4x7 decomposition.
        Split m=4 into 2+2.
        Each part is 3x2x7.
        Split 3x2x7 into:
          n=3 -> 2+1
          p=7 -> 2+2+2+1
        Blocks:
          (2x2) x (2x2) -> 2x2x2 (Strassen, rank 7) x 3 blocks
          (2x2) x (2x1) -> 2x2x1 (Schoolbook, rank 4) x 1 block
          (1x2) x (2x2) -> 1x2x2 (Schoolbook, rank 4) x 3 blocks
          (1x2) x (2x1) -> 1x2x1 (Schoolbook, rank 2) x 1 block
        Total rank for one 3x2x7: 3*7 + 4 + 3*4 + 2 = 21 + 4 + 12 + 2 = 39.
        Total rank for 3x4x7: 39 + 39 = 78.
        """
        n, m, p = 3, 4, 7
        
        # Accumulators
        # We will build the full U, V, W matrices of size (12x78), (28x78), (21x78)
        # But we don't know the final rank exactly until we sum up.
        # Let's just collect the blocks and concatenate.
        
        blocks = []
        
        def add_block(U_b, V_b, W_b, n_off, m_off, p_off):
            """
            U_b: (n_b*m_b) x rank_b
            V_b: (m_b*p_b) x rank_b
            W_b: (p_b*n_b) x rank_b
            n_off: offset in rows of A (i)
            m_off: offset in cols of A / rows of B (j)
            p_off: offset in cols of B (k)
            """
            blocks.append((U_b, V_b, W_b, n_off, m_off, p_off))
            
        # We process two main blocks: m_off=0 and m_off=2
        
        for m_main in [0, 2]:
            # Sub-blocks for n=3 (2+1) and p=7 (2+2+2+1)
            
            # n_part 0: rows 0,1 (size 2)
            # n_part 1: row 2 (size 1)
            
            # p_parts: cols 0,1 (size 2); 2,3 (size 2); 4,5 (size 2); 6 (size 1)
            
            # Block (2x2) x (2x2) for p_off 0, 2, 4
            for p_off in [0, 2, 4]:
                U_b, V_b, W_b = strassen_222()
                add_block(U_b, V_b, W_b, n_off=0, m_off=m_main, p_off=p_off)
                
            # Block (2x2) x (2x1) for p_off 6
            U_b, V_b, W_b = schoolbook_block(2, 2, 1)
            add_block(U_b, V_b, W_b, n_off=0, m_off=m_main, p_off=6)
            
            # Block (1x2) x (2x2) for p_off 0, 2, 4
            for p_off in [0, 2, 4]:
                U_b, V_b, W_b = schoolbook_block(1, 2, 2)
                add_block(U_b, V_b, W_b, n_off=2, m_off=m_main, p_off=p_off)
                
            # Block (1x2) x (2x1) for p_off 6
            U_b, V_b, W_b = schoolbook_block(1, 2, 1)
            add_block(U_b, V_b, W_b, n_off=2, m_off=m_main, p_off=6)

        # Now combine all blocks into full matrices
        total_rank = 0
        for U_b, V_b, W_b, _, _, _ in blocks:
            total_rank += U_b[0].__len__()
            
        U_full = [[0]*total_rank for _ in range(n*m)]
        V_full = [[0]*total_rank for _ in range(m*p)]
        W_full = [[0]*total_rank for _ in range(p*n)]
        
        col_offset = 0
        for U_b, V_b, W_b, n_off, m_off, p_off in blocks:
            n_b = len(U_b) // 2 if U_b else 0 # Hacky, better to pass dims
            # Actually, we can infer dims from offsets and sizes?
            # No, let's just iterate rows.
            
            # We need to know n_b, m_b, p_b to map indices correctly.
            # U_b is n_b * m_b rows.
            # V_b is m_b * p_b rows.
            # W_b is p_b * n_b rows.
            
            # We stored the blocks without explicit dims. Let's re-calculate or store them.
            # To be safe, let's just reconstruct the loop and store dims in the tuple.
            
            pass
        
        # Redo the accumulation with stored dimensions
        blocks = []
        
        def add_block(U_b, V_b, W_b, n_off, m_off, p_off, n_b, m_b, p_b):
            blocks.append((U_b, V_b, W_b, n_off, m_off, p_off, n_b, m_b, p_b))
            
        for m_main in [0, 2]:
            for p_off in [0, 2, 4]:
                U_b, V_b, W_b = strassen_222()
                add_block(U_b, V_b, W_b, 0, m_main, p_off, 2, 2, 2)
            
            U_b, V_b, W_b = schoolbook_block(2, 2, 1)
            add_block(U_b, V_b, W_b, 0, m_main, 6, 2, 2, 1)
            
            for p_off in [0, 2, 4]:
                U_b, V_b, W_b = schoolbook_block(1, 2, 2)
                add_block(U_b, V_b, W_b, 2, m_main, p_off, 1, 2, 2)
                
            U_b, V_b, W_b = schoolbook_block(1, 2, 1)
            add_block(U_b, V_b, W_b, 2, m_main, 6, 1, 2, 1)

        total_rank = 0
        for U_b, V_b, W_b, _, _, _, _, _, _ in blocks:
            total_rank += len(U_b[0])
            
        U_full = [[0]*total_rank for _ in range(n*m)]
        V_full = [[0]*total_rank for _ in range(m*p)]
        W_full = [[0]*total_rank for _ in range(p*n)]
        
        col_offset = 0
        for U_b, V_b, W_b, n_off, m_off, p_off, n_b, m_b, p_b in blocks:
            rank_b = len(U_b[0])
            
            for i in range(n_b):
                for j in range(m_b):
                    u_idx = i * m_b + j
                    u_global_idx = (n_off + i) * m + (m_off + j)
                    for c in range(rank_b):
                        U_full[u_global_idx][col_offset + c] = U_b[u_idx][c]
                        
            for j in range(m_b):
                for k in range(p_b):
                    v_idx = j * p_b + k
                    v_global_idx = (m_off + j) * p + (p_off + k)
                    for c in range(rank_b):
                        V_full[v_global_idx][col_offset + c] = V_b[v_idx][c]
                        
            for k in range(p_b):
                for i in range(n_b):
                    w_idx = k * n_b + i
                    w_global_idx = (p_off + k) * n + (n_off + i)
                    for c in range(rank_b):
                        W_full[w_global_idx][col_offset + c] = W_b[w_idx][c]
                        
            col_offset += rank_b
            
        return {"U": U_full, "V": V_full, "W": W_full, "rank": total_rank}

    # Try the optimized decomposition
    try:
        if time.time() < DEADLINE:
            result = build_full_decomp()
            # Verify
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return {"U": result["U"], "V": result["V"], "W": result["W"]}
            else:
                print("Verification failed for optimized decomposition, falling back to schoolbook.")
    except Exception as e:
        print(f"Error in optimized decomposition: {e}, falling back to schoolbook.")

    # Fallback to schoolbook
    result = schoolbook()
    return {"U": result["U"], "V": result["V"], "W": result["W"]}
# EVOLVE-BLOCK-END
