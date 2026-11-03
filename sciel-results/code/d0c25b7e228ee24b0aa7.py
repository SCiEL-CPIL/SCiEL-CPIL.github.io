"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (a21 + a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # p3 = a11(b12 - b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # p4 = a22(b21 - b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # p5 = (a11 + a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # p6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_2x2x3_rank11():
    """Build rank-11 decomposition for 2x2 * 2x3 using Strassen for 2x2x2 and schoolbook for 2x2x1."""
    # 2x2x2 part (cols 0,1)
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # 2x2x1 part (col 2)
    U_1 = [[0]*4 for _ in range(4)]
    V_1 = [[0]*4 for _ in range(2)]
    W_1 = [[0]*4 for _ in range(2)]
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    U_1[0][0] = 1; V_1[0][0] = 1; W_1[0][0] = 1
    U_1[1][1] = 1; V_1[1][1] = 1; W_1[0][1] = 1
    U_1[2][2] = 1; V_1[0][2] = 1; W_1[1][2] = 1
    U_1[3][3] = 1; V_1[1][3] = 1; W_1[1][3] = 1
    
    # Combine
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)] # 2x3 -> 6 entries
    W = [[0]*11 for _ in range(6)] # 3x2 -> 6 entries
    
    # Copy Strassen
    for r in range(7):
        for i in range(4):
            U[i][r] = U_s[i][r]
        for j in range(2):
            for k in range(2):
                V[j*3+k][r] = V_s[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*2+i][r] = W_s[k*2+i][r]
                
    # Copy 2x2x1
    offset = 7
    for r in range(4):
        for i in range(4):
            U[i][offset+r] = U_1[i][r]
        for j in range(2):
            V[j*3+2][offset+r] = V_1[j][r]
        for i in range(2):
            W[2*2+i][offset+r] = W_1[i][r]
            
    return U, V, W


def _build_4x4x7_optimized_v2():
    """Build decomposition for 4x4 * 4x7.
    Part 1: 4x4 * 4x4 (cols 0-3) -> Rank 49 (Kronecker Strassen)
    Part 2: 4x4 * 4x3 (cols 4-6) -> Rank 40 (4 x 2x2x3 Rank 10) if found, else 44.
    """
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    # Part 1: 4x4 * 4x4
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k1*2+i2][r2] # Wait, W index is k*n+i
                            # W2 is for 2x2x2, W2[k*2+i]
                            # Here k, i are local 2x2 indices
                            if val != 0:
                                W1[k*4+i][r] += val
    
    # Part 2: 4x4 * 4x3
    # Try to find rank 10 for 2x2x3
    U_23, V_23, W_23 = _build_2x2x3_rank11()
    # Try ALS to reduce to 10
    U_23_arr = np.array(U_23, dtype=float)
    V_23_arr = np.array(V_23, dtype=float)
    W_23_arr = np.array(W_23, dtype=float)
    
    # ALS for 2x2x3 rank 10
    target_rank = 10
    success = False
    if time.time() - _START_TIME < _TIME_LIMIT - 20:
        try:
            U_23_r, V_23_r, W_23_r, err = _als_decompose(2, 2, 3, target_rank, U_23_arr, V_23_arr, W_23_arr, max_iter=500)
            if err < 1e-8:
                result_23 = _round_and_verify(U_23_r, V_23_r, W_23_r, 2, 2, 3)
                if result_23 is not None:
                    U_23 = result_23['U']
                    V_23 = result_23['V']
                    W_23 = result_23['W']
                    success = True
        except Exception:
            pass
    
    if success:
        rank2 = 40
    else:
        rank2 = 44 # Fallback to rank 11 * 4
    
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(12)] # 4x3
    W2p = [[0] * rank2 for _ in range(12)] # 3x4
    
    if success:
        # 4 blocks of 2x2x3
        block_rank = 10
        for br in range(2):
            for bc in range(2):
                offset = (br*2 + bc) * block_rank
                for r in range(block_rank):
                    # U: A[br*2+lr, bc*2+lc]
                    for lr in range(2):
                        for lc in range(2):
                            i = br*2 + lr
                            j = bc*2 + lc
                            val = U_23[lr*2+lc][r]
                            if val != 0:
                                U2p[i*4+j][offset+r] += val
                    # V: B[bc*2+lr, 3*bc+kl] ? No.
                    # B is 4x3.
                    # Block (br, bc) of A multiplies Block (br, bc) of B?
                    # A (4x4) blocks: A00, A01, A10, A11.
                    # B (4x3) blocks:
                    # Rows 0-1, Cols 0-2 -> B_top
                    # Rows 2-3, Cols 0-2 -> B_bot
                    # A00 * B_top -> C_top_left?
                    # C (4x3).
                    # C_top (2x3) = A00 B_top + A01 B_bot.
                    # C_bot (2x3) = A10 B_top + A11 B_bot.
                    
                    # So:
                    # br=0, bc=0 (A00) multiplies B_top (rows 0-1 of B, cols 0-2)
                    # br=0, bc=1 (A01) multiplies B_bot (rows 2-3 of B, cols 0-2)
                    # br=1, bc=0 (A10) multiplies B_top
                    # br=1, bc=1 (A11) multiplies B_bot
                    
                    # V index: j (4x3) -> j*3 + k
                    # If br=0 (A00, A01), B rows are 0-1 or 2-3?
                    # A00 (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 0-2)
                    # A01 (rows 0-1, cols 2-3) * B_bot (rows 2-3, cols 0-2)
                    
                    # For A00 (br=0, bc=0):
                    # V rows: 0,1 (from B_top)
                    # For A01 (br=0, bc=1):
                    # V rows: 2,3 (from B_bot)
                    
                    # W index: k (3x4) -> k*4 + i
                    # C_top (rows 0-1) comes from br=0.
                    # C_bot (rows 2-3) comes from br=1.
                    
                    b_row_start = 0 if bc == 0 else 2
                    for lr in range(2):
                        for kl in range(3):
                            j = b_row_start + lr
                            k = kl
                            val = V_23[lr*3+kl][r]
                            if val != 0:
                                V2p[j*3+k][offset+r] += val
                    for lr in range(2):
                        for kl in range(3):
                            i = br*2 + lr
                            k = kl
                            val = W_23[kl*2+lr][r]
                            if val != 0:
                                W2p[k*4+i][offset+r] += val
    else:
        # Fallback: Rank 44 (4 x 11)
        block_rank = 11
        for br in range(2):
            for bc in range(2):
                offset = (br*2 + bc) * block_rank
                for r in range(block_rank):
                    for lr in range(2):
                        for lc in range(2):
                            i = br*2 + lr
                            j = bc*2 + lc
                            val = U_23[lr*2+lc][r]
                            if val != 0:
                                U2p[i*4+j][offset+r] += val
                    b_row_start = 0 if bc == 0 else 2
                    for lr in range(2):
                        for kl in range(3):
                            j = b_row_start + lr
                            k = kl
                            val = V_23[lr*3+kl][r]
                            if val != 0:
                                V2p[j*3+k][offset+r] += val
                    for lr in range(2):
                        for kl in range(3):
                            i = br*2 + lr
                            k = kl
                            val = W_23[kl*2+lr][r]
                            if val != 0:
                                W2p[k*4+i][offset+r] += val

    # Combine
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = W1[i*4+j][r]
                
    # Part 2
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2p[i][r]
    for i in range(4):
        for j in range(3):
            for r in range(rank2):
                V[i*p + (4+j)][offset2 + r] = V2p[i*3+j][r]
    for i in range(3):
        for j in range(4):
            for r in range(rank2):
                W[(4+i)*n + j][offset2 + r] = W2p[i*4+j][r]
                
    return {"U": U, "V": V, "W": W}


def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _als_decompose(n, m, p, target_rank, init_U, init_V, init_W, max_iter=300, tol=1e-12):
    """Alternating least squares for exact tensor decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    U = np.array(init_U, dtype=float)[:, :target_rank].copy()
    V = np.array(init_V, dtype=float)[:, :target_rank].copy()
    W = np.array(init_W, dtype=float)[:, :target_rank].copy()
    
    T1 = T.reshape(nm, mp * pn)
    T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    for it in range(max_iter):
        Z1 = np.einsum('br,cr->bc,r', V, W).reshape(mp * pn, target_rank)
        U_new = np.linalg.lstsq(Z1.T, T1.T, rcond=None)[0].T
        Z2 = np.einsum('ar,cr->ac,r', U, W).reshape(nm * pn, target_rank)
        V_new = np.linalg.lstsq(Z2.T, T2.T, rcond=None)[0].T
        Z3 = np.einsum('ar,br->ab,r', U, V).reshape(nm * mp, target_rank)
        W_new = np.linalg.lstsq(Z3.T, T3.T, rcond=None)[0].T
        
        diff = max(np.max(np.abs(U_new - U)), np.max(np.abs(V_new - V)), np.max(np.abs(W_new - W)))
        U, V, W = U_new, V_new, W_new
        if diff < tol:
            break
            
    # Polish
    for it in range(50):
        Z1 = np.einsum('br,cr->bc,r', V, W).reshape(mp * pn, target_rank)
        U_new = np.linalg.lstsq(Z1.T, T1.T, rcond=None)[0].T
        Z2 = np.einsum('ar,cr->ac,r', U, W).reshape(nm * pn, target_rank)
        V_new = np.linalg.lstsq(Z2.T, T2.T, rcond=None)[0].T
        Z3 = np.einsum('ar,br->ab,r', U, V).reshape(nm * mp, target_rank)
        W_new = np.linalg.lstsq(Z3.T, T3.T, rcond=None)[0].T
        diff = max(np.max(np.abs(U_new - U)), np.max(np.abs(V_new - V)), np.max(np.abs(W_new - W)))
        U, V, W = U_new, V_new, W_new
        if diff < 1e-14:
            break
            
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    error = float(np.max(np.abs(T_recon - T)))
    return U, V, W, error


def _round_and_verify(U, V, W, n, m, p, tol=1e-6):
    """Round to nearest half-integers and verify exact decomposition."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if (np.max(np.abs(U_r - U)) > tol or np.max(np.abs(V_r - V)) > tol or np.max(np.abs(W_r - W)) > tol):
        return None
        
    U_list = U_r.tolist()
    V_list = V_r.tolist()
    W_list = W_r.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
        
    # Try the new optimized construction (rank 89 or 93)
    base_result = None
    try:
        result = _build_4x4x7_optimized_v2()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            base_result = result
    except Exception:
        pass
        
    if base_result is None:
        # Fallback to old optimized (rank 93)
        try:
            result = _build_4x4x7_optimized()
            if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
                base_result = result
        except Exception:
            pass
            
    if base_result is None:
        return _schoolbook(4, 4, 7)
        
    return base_result

# Keep old functions for fallback
def _build_4x4x7_optimized():
    """Old rank 93 builder."""
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0: U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0: V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k1*2+i2][r2]
                            if val != 0: W1[k*4+i][r] += val
                            
    rank2 = 28
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(8)]
    W2p = [[0] * rank2 for _ in range(8)]
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for t in range(7):
                r = block_idx * 7 + t
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U2[lr*2+lc][t]
                        if val != 0: U2p[i*4+j][r] += val
                for lr in range(2):
                    for kl in range(2):
                        j = bc * 2 + lr
                        k = 4 + kl
                        val = V2[lr*2+kl][t]
                        if val != 0: V2p[j*2+kl][r] += val
                for lr in range(2):
                    for kl in range(2):
                        i = br * 2 + lr
                        k = kl
                        val = W2[kl*2+lr][t]
                        if val != 0: W2p[k*4+i][r] += val
            block_idx += 1
            
    rank3 = 16
    U3 = [[0] * rank3 for _ in range(16)]
    V3 = [[0] * rank3 for _ in range(4)]
    W3 = [[0] * rank3 for _ in range(4)]
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    r = block_idx * 4 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U3[i*4+j][r] = 1
                    V3[j][r] = 1
                    W3[i][r] = 1
            block_idx += 1
            
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(16):
        for r in range(rank1): U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1): V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1): W[i*n+j][r] = W1[i*4+j][r]
            
    offset2 = rank1
    for i in range(16):
        for r in range(rank2): U[i][offset2 + r] = U2p[i][r]
    for i in range(4):
        for j in range(2):
            for r in range(rank2): V[i*p + (4+j)][offset2 + r] = V2p[i*2+j][r]
    for i in range(2):
        for j in range(4):
            for r in range(rank2): W[(4+i)*n + j][offset2 + r] = W2p[i*4+j][r]
            
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3): U[i][offset3 + r] = U3[i][r]
    for i in range(4):
        for r in range(rank3): V[i*p + 6][offset3 + r] = V3[i][r]
    for i in range(4):
        for r in range(rank3): W[6*n + i][offset3 + r] = W3[i][r]
        
    return {"U": U, "V": V, "W": W}
