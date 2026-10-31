import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    a_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]],
    ]
    
    b_coeffs = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]],
    ]
    
    c_coeffs = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
    ]
    
    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            
            for S in range(2):
                for T in range(2):
                    if a_coeffs[s][S][T] == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            if a_coeffs[t][s2][t2] == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j][c_idx] += a_coeffs[s][S][T] * a_coeffs[t][s2][t2]
            
            for T in range(2):
                for K in range(2):
                    if b_coeffs[s][T][K] == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            if b_coeffs[t][t2][k2] == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k][c_idx] += b_coeffs[s][T][K] * b_coeffs[t][t2][k2]
            
            for K in range(2):
                for S in range(2):
                    if c_coeffs[s][S][K] == 0:
                        continue
                    for k2 in range(2):
                        for s2 in range(2):
                            if c_coeffs[t][s2][k2] == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * 4 + i][c_idx] += c_coeffs[s][S][K] * c_coeffs[t][s2][k2]
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    kept = list(range(49))
    
    # Multiple passes to find all dependencies
    for pass_num in range(3):
        if time.time() > deadline - 1.0:
            break
        changed = False
        for elim_idx in list(kept):
            other_cols = [c for c in kept if c != elim_idx]
            if not other_cols:
                continue
            
            U_other = U_arr[:, other_cols]
            V_other = V_arr[:, other_cols]
            W_other = W_arr[:, other_cols]
            
            U_target = U_arr[:, elim_idx]
            V_target = V_arr[:, elim_idx]
            W_target = W_arr[:, elim_idx]
            
            A_design = np.vstack([U_other, V_other, W_other])
            b_target = np.concatenate([U_target, V_target, W_target])
            
            try:
                coeffs, residuals, rank_ls, sv = np.linalg.lstsq(A_design, b_target, rcond=None)
                recon = A_design @ coeffs
                if np.allclose(recon, b_target, atol=1e-10):
                    kept.remove(elim_idx)
                    changed = True
            except:
                pass
        if not changed:
            break
    
    if not kept:
        return {"U": U, "V": V, "W": W}
    
    U_new = [[U[i][j] for j in kept] for i in range(16)]
    V_new = [[V[i][j] for j in kept] for i in range(16)]
    W_new = [[W[i][j] for j in kept] for i in range(16)]
    
    return {"U": U_new, "V": V_new, "W": W_new}
# EVOLVE-BLOCK-END
