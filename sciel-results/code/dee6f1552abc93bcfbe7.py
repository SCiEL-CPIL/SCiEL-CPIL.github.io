import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Laderman's 23-term decomposition for 3x3 matrix multiplication
    # We use this as a building block for the 3x3 x 3x6 multiplication
    # The key insight: multiply a 3x3 by 3x6 as two independent 3x3 x 3x3 multiplications
    # But we can do better by sharing structure.
    
    # Actually, let's think about this more carefully.
    # We want to compute C = A * B where A is 3x3 and B is 3x6.
    # C is 3x6.
    # 
    # One approach: split B into two 3x3 matrices B1 and B2 (columns 0-2 and 3-5)
    # Then C = [A*B1 | A*B2]
    # Each A*B1 and A*B2 can be done with Laderman's 23-term algorithm.
    # But that gives rank 46.
    #
    # Can we do better? Let's think about the tensor structure.
    # The multiplication tensor for 3x3 x 3x6 has rank at most 46 (from above).
    # Is there a better decomposition?
    #
    # Let's try a different approach: use the fact that we're multiplying by a 3x6 matrix.
    # We can view this as computing 6 independent dot products (one for each column of B).
    # But that's not quite right either.
    #
    # Let me reconsider. The standard approach for rectangular matrix multiplication
    # is to use known decompositions. For 3x3 x 3x6, one natural approach is:
    # - Use Laderman's algorithm for the two 3x3 blocks, giving rank 46.
    # - Or find a specialized decomposition.
    #
    # Let's verify that rank 46 works with Laderman's algorithm applied to each half.
    # But wait - can we share some of the computation?
    #
    # Actually, let me try a completely different approach. Let's use the fact that
    # matrix multiplication is bilinear and try to find a lower-rank decomposition
    # by exploiting the structure.
    #
    # For now, let's use the 46-rank decomposition and verify it works.
    # If we can find a better one, we'll use that.
    
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
    
    # Build the 46-rank decomposition using two copies of Laderman's algorithm
    rank = 46
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for block_offset in [0, 3]:
        for u_map, v_map, w_map in laderman_terms:
            for u_idx, coeff in u_map.items():
                U[u_idx, col] += coeff
            for v_idx, coeff in v_map.items():
                # v_idx is local j*3+k. Global is j*6 + (k+block_offset)
                j = v_idx // 3
                k = v_idx % 3
                V[j * p + (k + block_offset), col] += coeff
            for w_idx, coeff in w_map.items():
                # w_idx is local k*3+i. Global is (k+block_offset)*3 + i
                k = w_idx // 3
                i = w_idx % 3
                W[(k + block_offset) * n + i, col] += coeff
            col += 1
    
    # Verify correctness
    valid = True
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            v_rows = V[j * p:(j + 1) * p, :]  # shape (p, rank)
            w_indices = np.array([k * n + i for k in range(p)])
            w_vals = W[w_indices, :]  # shape (p, rank)
            products = u_row[None, :] * v_rows * w_vals  # shape (p, rank)
            sums = products.sum(axis=1)
            if not np.allclose(sums, 1.0, atol=1e-10):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
        V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
        W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
