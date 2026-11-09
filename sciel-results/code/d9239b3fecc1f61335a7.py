import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit

    def error_np(U, V, W):
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err

    def verify_exact(U, V, W):
        """Verify decomposition using exact rational arithmetic."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for t in range(r):
                        u_val = Fraction(U[i*m+j, t])
                        v_val = Fraction(V[j*p+k, t])
                        w_val = Fraction(W[k*n+i, t])
                        val += u_val * v_val * w_val
                    if val != 1:
                        return False
        return True

    # Schoolbook decomposition (rank 63)
    R0 = n * m * p
    U0 = np.zeros((n*m, R0))
    V0 = np.zeros((m*p, R0))
    W0 = np.zeros((p*n, R0))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, c] = 1
                V0[j*p+k, c] = 1
                W0[k*n+i, c] = 1
                c += 1

    best_R = R0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    def snap_half(x):
        return round(x * 2) / 2

    # ===== Strassen 2x2x2 decomposition (rank 7) =====
    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2 * 2x2 -> 2x2 (rank 7)
        A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)b11
        M3 = a11(b12-b22)
        M4 = a22(b21-b11)
        M5 = (a11+a12)b22
        M6 = (a21-a11)(b11+b12)
        M7 = (a12-a22)(b21+b22)
        
        C11 = M1+M4-M5+M7
        C12 = M3+M5
        C21 = M2+M4
        C22 = M1-M2+M3+M6
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a11+a22)(b11+b22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a21+a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a11(b12-b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = a22(b21-b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a11+a12)b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (a21-a11)(b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a12-a22)(b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # ===== Partitioned construction with Strassen for 2x2*2x2 block =====
    def build_partitioned():
        """
        Build 3x3*3x7 using block partitioning:
        Partition A (3x3) into [2; 1] rows and [2, 1] cols
        Partition B (3x7) into [2; 1] rows and [3, 4] cols
        
        Sub-problems:
        A11*B11: 2x2 * 2x3 (use Strassen-like or schoolbook rank 12)
        A12*B21: 2x1 * 1x3 (rank 3 schoolbook)
        A11*B12: 2x2 * 2x4 (rank 16 schoolbook)
        A12*B22: 2x1 * 1x4 (rank 4 schoolbook)
        A21*B11: 1x2 * 2x3 (rank 3 schoolbook)
        A22*B21: 1x1 * 1x3 (rank 3 schoolbook)
        A21*B12: 1x2 * 2x4 (rank 4 schoolbook)
        A22*B22: 1x1 * 1x4 (rank 4 schoolbook)
        """
        total_rank = 12 + 3 + 16 + 4 + 3 + 3 + 4 + 4  # = 49
        
        U = np.zeros((9, total_rank))
        V = np.zeros((21, total_rank))
        W = np.zeros((21, total_rank))
        
        col_offset = 0
        
        def add_block(n_sub, m_sub, p_sub, u_off, v_off, w_off):
            """Add a schoolbook decomposition block."""
            nonlocal col_offset
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U[u_off + i*m_sub + j, col_offset] = 1
                        V[v_off + j*p_sub + k, col_offset] = 1
                        W[w_off + k*n_sub + i, col_offset] = 1
                        col_offset += 1
            return col_offset
        
        # Block 1: A11*B11: 2x2 * 2x3
        # A rows 0-1, A cols 0-1; B rows 0-1, B cols 0-2
        # U: rows 0-3 (i*m+j for i in 0-1, j in 0-1)
        # V: rows 0-5 (j*p+k for j in 0-1, k in 0-2)
        # W: rows 0-5 (k*n+i for k in 0-2, i in 0-1)
        add_block(2, 2, 3, 0, 0, 0)
        
        # Block 2: A12*B21: 2x1 * 1x3
        # A rows 0-1, A col 2; B row 2, B cols 0-2
        # U: rows 2, 5 (i*3+2 for i in 0-1)
        # V: rows 6,7,8 (2*3+k for k in 0-2)
        # W: rows 0-5 (same as block 1)
        add_block(2, 1, 3, 2, 6, 0)
        
        # Block 3: A11*B12: 2x2 * 2x4
        # A rows 0-1, A cols 0-1; B rows 0-1, B cols 3-6
        # U: rows 0-3
        # V: rows 0-7 (j*4+k for j in 0-1, k in 0-3) -> but these are B cols 3-6
        #    V row index: j*p_full + k where p_full=7, so for B cols 3-6: j*7 + (3+k)
        #    For j=0: rows 3,4,5,6; for j=1: rows 10,11,12,13
        # W: rows 0-7 (k*n+i for k in 0-3, i in 0-1) -> C cols 3-6
        #    W row: k*3 + i where k is output col offset from 3, i in 0-1
        #    For k=0: rows 0,1; k=1: rows 3,4; k=2: rows 6,7; k=3: rows 9,10
        add_block(2, 2, 4, 0, 3, 0)  # V offset 3 because B cols start at 3
        
        # Block 4: A12*B22: 2x1 * 1x4
        # A rows 0-1, A col 2; B row 2, B cols 3-6
        # U: rows 2, 5
        # V: rows 10,11,12,13 (2*7 + 3+k for k in 0-3)
        # W: rows 0-7 (same as block 3)
        add_block(2, 1, 4, 2, 10, 0)
        
        # Block 5: A21*B11: 1x2 * 2x3
        # A row 2, A cols 0-1; B rows 0-1, B cols 0-2
        # U: rows 6,7 (2*3 + j for j in 0-1)
        # V: rows 0-5 (same as block 1)
        # W: rows 6,7,8 (2*3 + k for k in 0-2)
        add_block(1, 2, 3, 6, 0, 6)
        
        # Block 6: A22*B21: 1x1 * 1x3
        # A row 2, A col 2; B row 2, B cols 0-2
        # U: row 8
        # V: rows 6,7,8
        # W: rows 6,7,8
        add_block(1, 1, 3, 8, 6, 6)
        
        # Block 7: A21*B12: 1x2 * 2x4
        # A row 2, A cols 0-1; B rows 0-1, B cols 3-6
        # U: rows 6,7
        # V: rows 3,4,5,6,10,11,12,13
        # W: rows 9,10,11,12,13,14,15,16 (2*3 + (3+k) for k in 0-3)
        add_block(1, 2, 4, 6, 3, 9)
        
        # Block 8: A22*B22: 1x1 * 1x4
        # A row 2, A col 2; B row 2, B cols 3-6
        # U: row 8
        # V: rows 10,11,12,13
        # W: rows 9,10,11,12
        add_block(1, 1, 4, 8, 10, 9)
        
        return U, V, W, col_offset

    # Try partitioned construction
    if time.time() < deadline - 60:
        U_part, V_part, W_part, part_rank = build_partitioned()
        if error_np(U_part, V_part, W_part) < 1e-9:
            best_R = min(best_R, part_rank)
            best_U, best_V, best_W = U_part.copy(), V_part.copy(), W_part.copy()

    # Greedy column removal
    if best_R < R0 and time.time() < deadline - 30:
        U_temp = best_U.copy()
        V_temp = best_V.copy()
        W_temp = best_W.copy()
        current_R = best_R

        changed = True
        while changed:
            changed = False
            for col in range(current_R - 1, -1, -1):
                if time.time() > deadline - 25:
                    break
                
                U_bak = U_temp.copy()
                V_bak = V_temp.copy()
                W_bak = W_temp.copy()
                
                U_temp[:, col] = 0
                V_temp[:, col] = 0
                W_temp[:, col] = 0
                
                if error_np(U_temp, V_temp, W_temp) < 1e-10:
                    U_temp = np.delete(U_temp, col, axis=1)
                    V_temp = np.delete(V_temp, col, axis=1)
                    W_temp = np.delete(W_temp, col, axis=1)
                    current_R -= 1
                    changed = True
                else:
                    U_temp, V_temp, W_temp = U_bak, V_bak, W_bak

        if current_R < best_R:
            best_R = current_R
            best_U, best_V, best_W = U_temp.copy(), V_temp.copy(), W_temp.copy()

    # Local search to try to improve
    def local_search(U, V, W, max_iters=200):
        U = U.astype(float).copy()
        V = V.astype(float).copy()
        W = W.astype(float).copy()
        r = U.shape[1]
        best_local_err = error_np(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Try random perturbation
            choice = random.random()
            old_val = 0
            if choice < 0.33:
                idx = random.randint(0, n*m-1)
                col = random.randint(0, r-1)
                old_val = U[idx, col]
                U[idx, col] = snap_half(U[idx, col] + random.choice([-1, 1]) * 0.5)
            elif choice < 0.66:
                idx = random.randint(0, m*p-1)
                col = random.randint(0, r-1)
                old_val = V[idx, col]
                V[idx, col] = snap_half(V[idx, col] + random.choice([-1, 1]) * 0.5)
            else:
                idx = random.randint(0, p*n-1)
                col = random.randint(0, r-1)
                old_val = W[idx, col]
                W[idx, col] = snap_half(W[idx, col] + random.choice([-1, 1]) * 0.5)
            
            new_err = error_np(U, V, W)
            if new_err < best_local_err:
                best_local_err = new_err
                if best_local_err < 1e-10:
                    break
            else:
                # Revert
                if choice < 0.33:
                    U[idx, col] = old_val
                elif choice < 0.66:
                    V[idx, col] = old_val
                else:
                    W[idx, col] = old_val
        
        return U, V, W

    # Apply local search
    if time.time() < deadline - 20:
        U_ls, V_ls, W_ls = local_search(best_U, best_V, best_W, max_iters=300)
        if error_np(U_ls, V_ls, W_ls) < 1e-10:
            # Try to zero out more columns
            U_ls = U_ls.copy()
            V_ls = V_ls.copy()
            W_ls = W_ls.copy()
            cur_r = U_ls.shape[1]
            changed = True
            while changed and cur_r > 0:
                changed = False
                for col in range(cur_r - 1, -1, -1):
                    if time.time() > deadline - 15:
                        break
                    U_bak = U_ls.copy()
                    V_bak = V_ls.copy()
                    W_bak = W_ls.copy()
                    U_ls[:, col] = 0
                    V_ls[:, col] = 0
                    W_ls[:, col] = 0
                    if error_np(U_ls, V_ls, W_ls) < 1e-10:
                        U_ls = np.delete(U_ls, col, axis=1)
                        V_ls = np.delete(V_ls, col, axis=1)
                        W_ls = np.delete(W_ls, col, axis=1)
                        cur_r -= 1
                        changed = True
                    else:
                        U_ls, V_ls, W_ls = U_bak, V_bak, W_bak
                if cur_r < best_R:
                    best_R = cur_r
                    best_U, best_V, best_W = U_ls.copy(), V_ls.copy(), W_ls.copy()

    # Final verification
    if not verify_exact(best_U, best_V, best_W):
        # Fallback to schoolbook if verification fails
        R0 = n * m * p
        U0 = np.zeros((n*m, R0))
        V0 = np.zeros((m*p, R0))
        W0 = np.zeros((p*n, R0))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i*m+j, c] = 1
                    V0[j*p+k, c] = 1
                    W0[k*n+i, c] = 1
                    c += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
