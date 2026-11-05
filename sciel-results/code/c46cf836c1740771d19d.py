"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    # Strassen terms for 2x2x2
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # Output: C11 = M1 - M2 - M3 - M5, C12 = M3 + M5 + M7, C21 = M2 + M4 - M6, C22 = M1 - M4 + M6
    
    # Local indices:
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    # Format: list of (u_entries, v_entries, w_entries)
    # u_entries: list of ((i, j), val)
    # v_entries: list of ((j, k), val)
    # w_entries: list of ((i, k), val)
    
    strassen_decomp = [
        # M1
        (
            [((0, 0), 1), ((1, 1), 1)],
            [((0, 0), 1), ((1, 1), 1)],
            [((0, 0), 1), ((0, 1), 1), ((1, 0), -1), ((1, 1), -1)] # C11+1, C12+1, C21-1, C22-1? 
            # Wait, M1 contributes to C11 (+1), C22 (+1).
            # Let's re-verify the Strassen combination for W.
            # C11 = M1 - M2 - M3 - M5
            # C12 = M3 + M5 + M7
            # C21 = M2 + M4 - M6
            # C22 = M1 - M4 + M6
        )
    ]
    
    # The previous code hardcoded the W entries based on the final C.
    # It is safer to stick to the proven coefficients from the seed program's logic but re-implement the assignment.
    
    def get_strassen_terms():
        # Returns list of (U_local, V_local, W_local)
        # U_local: dict {(i,j): val}
        # V_local: dict {(j,k): val}
        # W_local: dict {(i,k): val}
        
        terms = []
        
        # M1 = (a00+a11)*(b00+b11)
        u = {(0,0): 1, (1,1): 1}
        v = {(0,0): 1, (1,1): 1}
        w = {(0,0): 1, (1,1): 1} # M1 adds to C00 and C11
        terms.append((u, v, w))
        
        # M2 = (a10+a11)*b00
        u = {(1,0): 1, (1,1): 1}
        v = {(0,0): 1}
        w = {(1,0): 1, (1,1): -1} # C21=M2+... => +1, C22=M1-M4+M6... wait.
        # C21 = M2 + M4 - M6. So M2 adds +1 to C21.
        # C22 = M1 - M4 + M6. M2 does not appear.
        # C11 = M1 - M2 - M3 - M5. M2 adds -1 to C11.
        # C12 = M3 + M5 + M7. M2 does not appear.
        # So W for M2: C21:+1, C11:-1.
        w = {(1,0): 1, (0,0): -1}
        terms.append((u, v, w))
        
        # M3 = a00*(b01-b11)
        u = {(0,0): 1}
        v = {(0,1): 1, (1,1): -1}
        # C12 = M3 + M5 + M7 => M3 adds +1 to C12.
        # C11 = ... - M3 ... => M3 adds -1 to C11.
        w = {(0,1): 1, (0,0): -1}
        terms.append((u, v, w))
        
        # M4 = a11*(b10-b00)
        u = {(1,1): 1}
        v = {(1,0): 1, (0,0): -1}
        # C21 = M2 + M4 - M6 => M4 adds +1 to C21.
        # C22 = M1 - M4 + M6 => M4 adds -1 to C22.
        w = {(1,0): 1, (1,1): -1}
        terms.append((u, v, w))
        
        # M5 = (a00+a01)*b11
        u = {(0,0): 1, (0,1): 1}
        v = {(1,1): 1}
        # C12 = M3 + M5 + M7 => M5 adds +1 to C12.
        # C11 = ... - M5 ... => M5 adds -1 to C11.
        w = {(0,1): 1, (0,0): -1}
        terms.append((u, v, w))
        
        # M6 = (a10-a00)*(b00+b01)
        u = {(1,0): 1, (0,0): -1}
        v = {(0,0): 1, (0,1): 1}
        # C21 = M2 + M4 - M6 => M6 adds -1 to C21.
        # C22 = M1 - M4 + M6 => M6 adds +1 to C22.
        w = {(1,0): -1, (1,1): 1}
        terms.append((u, v, w))
        
        # M7 = (a01-a11)*(b10+b11)
        u = {(0,1): 1, (1,1): -1}
        v = {(1,0): 1, (1,1): 1}
        # C12 = M3 + M5 + M7 => M7 adds +1 to C12.
        w = {(0,1): 1}
        terms.append((u, v, w))
        
        return terms

    ST = get_strassen_terms()

    total_rank = 44
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    def apply_strassen_block(i0, j0, k0, start_col):
        for r in range(7):
            u_dict, v_dict, w_dict = ST[r]
            col = start_col + r
            
            for (li, lj), val in u_dict.items():
                U[(i0 + li) * m + (j0 + lj), col] = val
            
            for (lj, lk), val in v_dict.items():
                V[(j0 + lj) * p + (k0 + lk), col] = val
            
            for (li, lk), val in w_dict.items():
                W[(k0 + lk) * n + (i0 + li), col] = val
        
        return start_col + 7

    def apply_schoolbook_block_1x2(i0, j0, k0, start_col):
        # A[i0, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0, k0:k0+2]
        # 4 terms
        for t in range(4):
            col = start_col + t
            j_local = t % 2
            k_local = t // 2
            
            U[i0 * m + (j0 + j_local), col] = 1.0
            V[(j0 + j_local) * p + (k0 + k_local), col] = 1.0
            W[(k0 + k_local) * n + i0, col] = 1.0
        
        return start_col + 4

    col = 0
    # Top 2 rows of A (i=0,1), full 4x4 of B
    # Split B into 2x2 blocks
    # Block (0,0) of B corresponds to rows 0-1, cols 0-1
    col = apply_strassen_block(0, 0, 0, col)
    # Block (0,2) of B corresponds to rows 0-1, cols 2-3
    col = apply_strassen_block(0, 0, 2, col)
    # Block (2,0) of B corresponds to rows 2-3, cols 0-1
    col = apply_strassen_block(0, 2, 0, col)
    # Block (2,2) of B corresponds to rows 2-3, cols 2-3
    col = apply_strassen_block(0, 2, 2, col)
    
    # Bottom row of A (i=2), full 4x4 of B
    # Split B into 2x2 blocks, but A row is 1x4, so each block is 1x2 * 2x2
    col = apply_schoolbook_block_1x2(2, 0, 0, col)
    col = apply_schoolbook_block_1x2(2, 0, 2, col)
    col = apply_schoolbook_block_1x2(2, 2, 0, col)
    col = apply_schoolbook_block_1x2(2, 2, 2, col)

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verification
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    try:
        np.random.seed(12345)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        C_comp = np.zeros((n, p))
        for r in range(total_rank):
            ar = 0.0
            for i in range(n):
                for j in range(m):
                    ar += U[i * m + j, r] * A[i, j]
            
            br = 0.0
            for j in range(m):
                for k in range(p):
                    br += V[j * p + k, r] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i, r] * ar * br

        max_err = float(np.max(np.abs(C_comp - C_expected)))
        if max_err > 1e-9:
            U_sb, V_sb, W_sb = schoolbook()
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    except Exception:
        U_sb, V_sb, W_sb = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
