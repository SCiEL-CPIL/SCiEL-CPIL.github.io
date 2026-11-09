import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Laderman's 23-term decomposition for 3x3 matrix multiplication
    # Indices: A: i*3+j, B: j*3+k, C: k*3+i
    # We store as dicts for clarity, then convert to arrays
    
    laderman_terms = [
        ({0: 1}, {1: 1, 4: -1}, {3: 1, 6: -1}),
        ({0: 1, 1: 1}, {4: 1}, {3: 1, 4: 1}),
        ({4: 1, 5: 1}, {8: 1}, {7: 1, 8: 1}),
        ({8: 1}, {6: 1, 0: -1}, {2: 1, 0: -1}),
        ({6: 1, 8: 1}, {0: 1}, {2: 1, 0: 1}),
        ({2: 1, 8: 1}, {6: 1}, {0: 1, 2: 1}),
        ({0: 1, 4: -1}, {4: 1, 8: 1}, {3: 1, 6: 1, 4: -1, 7: -1}),
        ({4: 1, 8: -1}, {8: 1, 0: 1}, {7: 1, 1: 1, 8: -1, 2: -1}),
        ({8: 1, 0: -1}, {0: 1, 4: 1}, {2: 1, 5: 1, 0: -1, 3: -1}),
        ({1: 1, 4: 1}, {5: 1}, {6: 1, 7: 1}),
        ({4: 1, 7: 1}, {7: 1}, {4: 1, 5: 1}),
        ({7: 1, 1: 1}, {4: 1}, {5: 1, 3: 1}),
        ({2: 1, 5: 1}, {6: 1}, {0: 1, 1: 1}),
        ({5: 1, 8: 1}, {2: 1}, {7: 1, 8: 1}),
        ({8: 1, 2: 1}, {6: 1}, {0: 1, 2: 1}),
        ({0: 1, 3: 1}, {1: 1}, {3: 1, 4: 1}),
        ({3: 1, 6: 1}, {5: 1}, {6: 1, 7: 1}),
        ({6: 1, 0: 1}, {7: 1}, {1: 1, 2: 1}),
        ({1: 1, 3: 1}, {4: 1}, {3: 1, 4: 1}),
        ({4: 1, 6: 1}, {7: 1}, {6: 1, 7: 1}),
        ({7: 1, 0: 1}, {1: 1}, {1: 1, 2: 1}),
        ({2: 1, 3: 1}, {6: 1}, {0: 1, 1: 1}),
        ({5: 1, 6: 1}, {2: 1}, {7: 1, 8: 1}),
    ]
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block_offset in [0, 3]:
        for u_map, v_map, w_map in laderman_terms:
            for u_idx, coeff in u_map.items():
                U[u_idx][col] += coeff
            for v_idx, coeff in v_map.items():
                # v_idx is local j*3+k. Global is j*6 + (k+block_offset)
                # v_idx = j*3+k -> j = v_idx//3, k = v_idx%3
                j = v_idx // 3
                k = v_idx % 3
                V[j * p + (k + block_offset)][col] += coeff
            for w_idx, coeff in w_map.items():
                # w_idx is local k*3+i. Global is (k+block_offset)*3 + i
                # w_idx = k*3+i -> k = w_idx//3, i = w_idx%3
                k = w_idx // 3
                i = w_idx % 3
                W[(k + block_offset) * n + i][col] += coeff
            col += 1
            
    # Verify correctness using vectorized operations
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    valid = True
    for i in range(n):
        for j in range(m):
            # For each k, compute sum_t U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
            u_row = U_arr[i * m + j, :]  # shape (rank,)
            v_col = V_arr[j * p:(j + 1) * p, :]  # shape (p, rank)
            # We need W[k*n + i] for k in 0..p-1
            w_indices = np.array([k * n + i for k in range(p)])
            w_vals = W_arr[w_indices, :]  # shape (p, rank)
            # Sum over t for each k
            products = u_row * v_col * w_vals  # shape (p, rank)
            sums = products.sum(axis=1)  # shape (p,)
            if not np.allclose(sums, 1.0, atol=1e-10):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return {"U": U, "V": V, "W": W}
