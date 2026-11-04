"""Fresh program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

def find_small_decomposition(n_sub, m_sub, p_sub, rank_target, time_budget, rng):
    """
    Find a rank-rank_target decomposition for n_sub x m_sub x p_sub multiplication.
    Returns U, V, W as lists of lists, or None.
    """
    start_time = time.time()
    NM_s = n_sub * m_sub
    MP_s = m_sub * p_sub
    PN_s = p_sub * n_sub
    
    # Target tensor support
    support = [(i * m_sub + j, j * p_sub + k, k * n_sub + i) 
               for i in range(n_sub) for j in range(m_sub) for k in range(p_sub)]
    
    def compute_error(U, V, W):
        err = 0.0
        for (a, b, c) in support:
            val = 0.0
            for r in range(rank_target):
                val += U[a][r] * V[b][r] * W[c][r]
            err += (val - 1.0) ** 2
        return err

    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    # Try a few restarts
    for restart in range(10):
        if time.time() > start_time + time_budget:
            break
            
        # Initialize with small integers
        U = rng.integers(-1, 2, size=(NM_s, rank_target)).astype(float)
        V = rng.integers(-1, 2, size=(MP_s, rank_target)).astype(float)
        W = rng.integers(-1, 2, size=(PN_s, rank_target)).astype(float)
        
        err = compute_error(U, V, W)
        
        # Local search: coordinate descent
        for _ in range(200):
            if time.time() > start_time + time_budget:
                break
            improved = False
            # Try adjusting U
            for a in range(NM_s):
                for r in range(rank_target):
                    old = U[a, r]
                    for new in [old - 1, old + 1]:
                        if new == old: continue
                        U[a, r] = new
                        new_err = compute_error(U, V, W)
                        if new_err < err - 1e-9:
                            err = new_err
                            improved = True
                        else:
                            U[a, r] = old
            # Try adjusting V
            for b in range(MP_s):
                for r in range(rank_target):
                    old = V[b, r]
                    for new in [old - 1, old + 1]:
                        if new == old: continue
                        V[b, r] = new
                        new_err = compute_error(U, V, W)
                        if new_err < err - 1e-9:
                            err = new_err
                            improved = True
                        else:
                            V[b, r] = old
            # Try adjusting W
            for c in range(PN_s):
                for r in range(rank_target):
                    old = W[c, r]
                    for new in [old - 1, old + 1]:
                        if new == old: continue
                        W[c, r] = new
                        new_err = compute_error(U, V, W)
                        if new_err < err - 1e-9:
                            err = new_err
                            improved = True
                        else:
                            W[c, r] = old
                            
            if err < 1e-9:
                break
            if not improved:
                break
                
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
    if best_err < 1e-9:
        return best_U.astype(int).tolist(), best_V.astype(int).tolist(), best_W.astype(int).tolist()
    return None

def construct():
    start_time = time.time()
    time_limit = 145
    
    # Strategy: Decompose 2x3x6 into 3 blocks of 2x3x2.
    # Rank of 2x3x2 is 7. Total Rank = 21.
    
    rng = np.random.default_rng(42)
    
    # Find decomposition for 2x3x2
    sub_result = find_small_decomposition(2, 3, 2, 7, 10, rng)
    
    if sub_result:
        U_sub, V_sub, W_sub = sub_result
        # Assemble for 2x3x6
        # B is split into 3 blocks of 3x2: B1, B2, B3
        # C is split into 3 blocks of 2x2: C1, C2, C3
        # C1 = A B1, C2 = A B2, C3 = A B3
        
        rank_total = 21
        U = [[0] * rank_total for _ in range(6)] # 2x3
        V = [[0] * rank_total for _ in range(18)] # 3x6
        W = [[0] * rank_total for _ in range(12)] # 6x2
        
        # Block 1: B cols 0,1 -> C cols 0,1
        # U is same for all blocks (A is shared)
        # V is different for each block
        # W is different for each block
        
        for r in range(7):
            # U part
            for i in range(6):
                U[i][r] = U_sub[i][r]
                
            # V part (Block 1: rows 0-2, cols 0-1 of B)
            # V_sub index: j*p_sub + k. p_sub=2.
            # B index: j*6 + k.
            # Block 1: k=0,1.
            for j in range(3):
                for k in range(2):
                    v_sub_idx = j * 2 + k
                    v_main_idx = j * 6 + k
                    V[v_main_idx][r] = V_sub[v_sub_idx][r]
                    
            # W part (Block 1: rows 0-1, cols 0-1 of C)
            # W_sub index: k*n_sub + i. n_sub=2.
            # C index: k*2 + i.
            # Block 1: k=0,1.
            for k in range(2):
                for i in range(2):
                    w_sub_idx = k * 2 + i
                    w_main_idx = k * 2 + i
                    W[w_main_idx][r] = W_sub[w_sub_idx][r]
                    
        for r in range(7, 14):
            r_sub = r - 7
            # U part
            for i in range(6):
                U[i][r] = U_sub[i][r_sub]
                
            # V part (Block 2: rows 0-2, cols 2-3 of B)
            for j in range(3):
                for k in range(2):
                    v_sub_idx = j * 2 + k
                    v_main_idx = j * 6 + (2 + k)
                    V[v_main_idx][r] = V_sub[v_sub_idx][r_sub]
                    
            # W part (Block 2: rows 2-3, cols 0-1 of C)
            for k in range(2):
                for i in range(2):
                    w_sub_idx = k * 2 + i
                    w_main_idx = (2 + k) * 2 + i
                    W[w_main_idx][r] = W_sub[w_sub_idx][r_sub]
                    
        for r in range(14, 21):
            r_sub = r - 14
            # U part
            for i in range(6):
                U[i][r] = U_sub[i][r_sub]
                
            # V part (Block 3: rows 0-2, cols 4-5 of B)
            for j in range(3):
                for k in range(2):
                    v_sub_idx = j * 2 + k
                    v_main_idx = j * 6 + (4 + k)
                    V[v_main_idx][r] = V_sub[v_sub_idx][r_sub]
                    
            # W part (Block 3: rows 4-5, cols 0-1 of C)
            for k in range(2):
                for i in range(2):
                    w_sub_idx = k * 2 + i
                    w_main_idx = (4 + k) * 2 + i
                    W[w_main_idx][r] = W_sub[w_sub_idx][r_sub]
                    
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: Schoolbook Rank 36
        rank = n * m * p
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
