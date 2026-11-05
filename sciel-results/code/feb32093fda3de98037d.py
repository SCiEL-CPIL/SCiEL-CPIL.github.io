import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0

    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        i_idx, j_idx, k_idx = np.indices((n, m, p))
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        U_vals = U[u_idx, :]
        V_vals = V[v_idx, :]
        W_vals = W[w_idx, :]
        
        total = np.sum(U_vals * V_vals * W_vals, axis=-1)
        
        return np.all(np.abs(total - 1.0) < tol)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_block_decomp(partition):
        terms = []
        
        def add_strassen_2x2x2(r1, c1, k1):
            A00 = r1 * m + c1
            A01 = r1 * m + (c1 + 1)
            A10 = (r1 + 1) * m + c1
            A11 = (r1 + 1) * m + (c1 + 1)
            
            B00 = c1 * p + k1
            B01 = c1 * p + (k1 + 1)
            B10 = (c1 + 1) * p + k1
            B11 = (c1 + 1) * p + (k1 + 1)
            
            C00_w = k1 * n + r1
            C01_w = (k1 + 1) * n + r1
            C10_w = k1 * n + (r1 + 1)
            C11_w = (k1 + 1) * n + (r1 + 1)
            
            terms.append(({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1}))
            terms.append(({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1}))
            terms.append(({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1}))
            terms.append(({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1}))
            terms.append(({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1}))
            terms.append(({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1}))
            terms.append(({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1}))

        for r1, r2 in partition:
            for c1, c2 in partition:
                for k1, k2 in partition:
                    ar = r2 - r1
                    ac = c2 - c1
                    bc = k2 - k1
                    
                    if ar == 2 and ac == 2 and bc == 2:
                        add_strassen_2x2x2(r1, c1, k1)
                    else:
                        # For non-2x2x2 blocks, we can use Rank 1 if the block is "simple"
                        # Actually, we can always use Rank 1 for any block by setting U=1, V=1, W=1
                        # But wait, if the block is 4x4x4, Rank 1 is not sufficient for MM tensor?
                        # If the tensor is all-ones, Rank 1 is sufficient.
                        # If the tensor is MM tensor, Rank 1 is NOT sufficient for 4x4x4.
                        # The MM tensor for 4x4x4 has rank 49.
                        # So we MUST use Strassen for 4x4x4?
                        # No, Strassen gives 49.
                        # If we use Schoolbook for 4x4x4, rank is 64.
                        # So we should use Strassen for 4x4x4.
                        
                        # But the current code only handles 2x2x2 Strassen.
                        # We need to handle 4x4x4 Strassen or use Schoolbook.
                        # For now, let's use Schoolbook for non-2x2x2 to be safe, 
                        # but we know 4x4x4 can be 49.
                        
                        # Let's implement 4x4x4 Strassen?
                        # Or just use Schoolbook for 4x4x4 (Rank 64) and Rank 1 for others.
                        # Rank 64 + 7*1 = 71.
                        # This is better than 117.
                        
                        for ii in range(r1, r2):
                            for jj in range(c1, c2):
                                for kk in range(k1, k2):
                                    u_idx = ii * m + jj
                                    v_idx = jj * p + kk
                                    w_idx = kk * n + ii
                                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx, r] = val
            for idx, val in V_dict.items():
                V[idx, r] = val
            for idx, val in W_dict.items():
                W[idx, r] = val
        
        return U, V, W

    partitions = [
        [(0, 4), (4, 5)],  # [4, 1]
        [(0, 2), (2, 4), (4, 5)],  # [2, 2, 1]
    ]
    
    best_U, best_V, best_W = get_schoolbook()
    best_rank = best_U.shape[1]
    
    for partition in partitions:
        if time.time() > deadline - 20.0:
            break
        U, V, W = build_block_decomp(partition)
        rank = U.shape[1]
        if verify(U, V, W):
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
