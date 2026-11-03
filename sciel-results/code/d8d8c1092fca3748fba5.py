import time
import numpy as np

_TIME_LIMIT = 120
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    result = np.zeros((n, m, p))
    for c in range(rank):
        uc = U_arr[:, c]
        vc = V_arr[:, c]
        wc = W_arr[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    return np.allclose(result, 1.0, atol=1e-9)

def _find_3x3x2_decomp(target_rank):
    n, m, p = 3, 3, 2
    for _ in range(10000):
        if _time_left() < 10:
            break
        U = np.random.choice([0, 1, -1], size=(n*m, target_rank))
        V = np.random.choice([0, 1, -1], size=(m*p, target_rank))
        W = np.random.choice([0, 1, -1], size=(p*n, target_rank))
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U, V, W
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = 3, 3, 8
    
    # Fallback: Schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find 3x3x2 decomposition
    for target_rank in [15, 16, 17, 18]:
        if _time_left() < 20:
            break
        decomp = _find_3x3x2_decomp(target_rank)
        if decomp is not None:
            U_332, V_332, W_332 = decomp
            # Tile 4 times for 3x3x8
            # U is same for all 4 blocks? No, U is for A, which is shared.
            # Wait, the tensor for 3x3x8 is not simply 4 copies of 3x3x2.
            # A is shared.
            # The decomposition is a sum of 4 tensors, each corresponding to 2 columns of B.
            # U is the same for all 4 blocks (since A is the same).
            # V is different for each block (different columns of B).
            # W is different for each block (different columns of C).
            
            # So we need to construct U, V, W for the full 3x3x8.
            # U_full: 9 x (4 * target_rank)
            # V_full: 24 x (4 * target_rank)
            # W_full: 24 x (4 * target_rank)
            
            # Actually, U is repeated 4 times? No.
            # The tensor is sum over c of U[i,j,c] V[j,k,c] W[k,i,c].
            # For the 4 blocks, we have 4 sets of indices.
            # Block b (b=0..3) corresponds to columns 2b, 2b+1 of B and C.
            # U_block_b is the same U_332 for all b.
            # V_block_b is V_332 with indices mapped to columns 2b, 2b+1.
            # W_block_b is W_332 with indices mapped to columns 2b, 2b+1.
            
            # So U_full has 4 copies of U_332? No, U is shared.
            # Wait, if U is shared, then U_full is just U_332 repeated?
            # No, the rank is the sum of the ranks of the blocks if they are independent.
            # But U is the same, so we can't just sum the ranks.
            # Actually, the tensor is T = T_0 + T_1 + T_2 + T_3, where T_b is the tensor for block b.
            # T_b[i,j,k] = U_332[i,j,c] V_332[j, k', c] W_332[k', i, c] where k' is the local column index.
            # So the full decomposition is the sum of 4 decompositions.
            # The rank is 4 * target_rank.
            # U_full: column c of block b is U_332[:, c].
            # So U_full has 4 * target_rank columns, each being a copy of U_332[:, c].
            # V_full: column c of block b is V_332[:, c] mapped to global columns.
            # W_full: column c of block b is W_332[:, c] mapped to global columns.
            
            U_full = np.zeros((n*m, 4 * target_rank))
            V_full = np.zeros((m*p, 4 * target_rank))
            W_full = np.zeros((p*n, 4 * target_rank))
            
            for b in range(4):
                for c in range(target_rank):
                    col = b * target_rank + c
                    U_full[:, col] = U_332[:, c]
                    # V_332 is 6 x target_rank, indices j*p+k' where p=2, k' in 0,1
                    # Global k = 2b + k'
                    for j in range(m):
                        for kp in range(2):
                            V_full[j*p + 2*b + kp, col] = V_332[j*2 + kp, c]
                    # W_332 is 6 x target_rank, indices k'*n+i where p=2, k' in 0,1
                    # Global k = 2b + k'
                    for kp in range(2):
                        for i in range(n):
                            W_full[(2*b + kp)*n + i, col] = W_332[kp*3 + i, c]
            
            if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    return best_result
