import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    def find_224_decomposition(max_time=30):
        """Find a rank-12 decomposition for 2x2x4 matmul."""
        rank = 12
        U_size = 4
        V_size = 8
        W_size = 8
        
        np.random.seed(42)
        
        best_error = float('inf')
        best_UVW = None
        
        # Try multiple random initializations with damped fixed-point iteration
        for attempt in range(200):
            if time.time() > deadline - max_time:
                break
            
            # Random initialization with various scales
            scale = np.random.choice([0.5, 1.0, 2.0])
            U = np.random.randn(U_size, rank) * scale
            V = np.random.randn(V_size, rank) * scale
            W = np.random.randn(W_size, rank) * scale
            
            for iteration in range(500):
                if time.time() > deadline - max_time:
                    break
                
                # Damped updates
                W_new = W.copy()
                for k in range(4):
                    for i in range(2):
                        for r in range(rank):
                            s = sum(U[i*2+j, r] * V[j*4+k, r] for j in range(2))
                            if abs(s) > 1e-10:
                                W_new[k*2+i, r] = 1.0 / s
                W = 0.5 * W + 0.5 * W_new
                
                U_new = U.copy()
                for i in range(2):
                    for j in range(2):
                        for r in range(rank):
                            s = sum(V[j*4+k, r] * W[k*2+i, r] for k in range(4))
                            if abs(s) > 1e-10:
                                U_new[i*2+j, r] = 1.0 / s
                U = 0.5 * U + 0.5 * U_new
                
                V_new = V.copy()
                for j in range(2):
                    for k in range(4):
                        for r in range(rank):
                            s = sum(U[i*2+j, r] * W[k*2+i, r] for i in range(2))
                            if abs(s) > 1e-10:
                                V_new[j*4+k, r] = 1.0 / s
                V = 0.5 * V + 0.5 * V_new
                
                # Check error
                err = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            val = sum(U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r] for r in range(rank))
                            err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    best_UVW = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-14:
                    break
            
            if best_error < 1e-14:
                break
        
        if best_UVW is None or best_error > 1e-8:
            return None
        
        U, V, W = best_UVW
        
        # Round to nearest half-integer
        for rounding in [(2, 2), (1, 1)]:
            factor = rounding[0]
            U_rounded = np.round(U * factor) / factor
            V_rounded = np.round(V * factor) / factor
            W_rounded = np.round(W * factor) / factor
            
            err = 0
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        val = sum(U_rounded[i*2+j, r] * V_rounded[j*4+k, r] * W_rounded[k*2+i, r] for r in range(rank))
                        err += (val - 1.0) ** 2
            
            if err < 1e-10:
                return (U_rounded, V_rounded, W_rounded)
        
        return None

    # Try to find rank-12 decomposition for 2x2x4
    decomp_224 = find_224_decomposition(max_time=40)
    
    if decomp_224 is not None:
        U_224, V_224, W_224 = decomp_224
        rank = 48
        U = [[0] * rank for _ in range(n * m)]  # 8 x 48
        V = [[0] * rank for _ in range(m * p)]  # 32 x 48
        W = [[0] * rank for _ in range(p * n)]  # 16 x 48
        
        def add_224_block(base_rank, i_offset, j_offset, k_offset):
            for r in range(12):
                current_rank = base_rank + r
                for li in range(2):
                    for lj in range(2):
                        val = U_224[li*2+lj, r]
                        if val != 0:
                            U[(li + i_offset) * 4 + (lj + j_offset)][current_rank] += val
                for lj in range(2):
                    for lk in range(4):
                        val = V_224[lj*4+lk, r]
                        if val != 0:
                            V[(lj + j_offset) * 8 + (lk + k_offset)][current_rank] += val
                for lk in range(4):
                    for li in range(2):
                        val = W_224[lk*2+li, r]
                        if val != 0:
                            W[(lk + k_offset) * 2 + (li + i_offset)][current_rank] += val
        
        add_224_block(0, 0, 0, 0)
        add_224_block(12, 0, 0, 4)
        add_224_block(24, 0, 2, 0)
        add_224_block(36, 0, 2, 4)
        
        # Verify
        def verify_full():
            for i in range(2):
                for j in range(4):
                    for k in range(8):
                        total = 0
                        for r in range(rank):
                            u_idx = i * 4 + j
                            v_idx = j * 8 + k
                            w_idx = k * 2 + i
                            total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                        if abs(total - 1) > 1e-10:
                            return False
            return True
        
        if verify_full():
            return {"U": U, "V": V, "W": W}
    
    # Fallback: Strassen-based rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    def add_strassen_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            cr = base_rank + r
            for idx, val in strassen_U[r]:
                li, lj = divmod(idx, 2)
                U[(li + i_off) * 4 + (lj + j_off)][cr] += val
            for idx, val in strassen_V[r]:
                lj, lk = divmod(idx, 2)
                V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
            for idx, val in strassen_W[r]:
                lk, li = divmod(idx, 2)
                W[(lk + k_off) * 2 + (li + i_off)][cr] += val
    
    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(7, 0, 0, 2)
    add_strassen_block(14, 0, 0, 4)
    add_strassen_block(21, 0, 0, 6)
    add_strassen_block(28, 0, 2, 0)
    add_strassen_block(35, 0, 2, 2)
    add_strassen_block(42, 0, 2, 4)
    add_strassen_block(49, 0, 2, 6)
    
    # Verify the Strassen decomposition
    def verify_strassen():
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    total = 0
                    for r in range(rank):
                        u_idx = i * 4 + j
                        v_idx = j * 8 + k
                        w_idx = k * 2 + i
                        total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    if verify_strassen():
        return {"U": U, "V": V, "W": W}
    
    # Last resort: trivial rank 64
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
