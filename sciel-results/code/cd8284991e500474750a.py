import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_schoolbook

    def verify_decomposition(U, V, W):
        n, m, p = DIMENSIONS
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def add_strassen_block(U, V, W, start_col, k_offset):
        """
        Add a 2x2x2 Strassen block for columns k_offset and k_offset+1.
        Returns the number of columns used (7).
        """
        # Strassen products for 2x2 * 2x2
        # M1 = A00 * (B01 - B11)
        # M2 = (A00 + A10) * B11
        # M3 = (A11 - A01) * B00
        # M4 = A11 * (B00 + B10)
        # M5 = (A00 + A11) * (B10 + B11)
        # M6 = (A01 - A11) * (B00 + B10)
        # M7 = (A00 - A01) * (B01 + B11)
        
        # Outputs:
        # C00 = M5 - M4 + M3 + M7
        # C01 = M1 + M4
        # C10 = M2 + M3
        # C11 = M1 - M5 + M2 + M6
        
        # Indices:
        # A is 2x2: A[0,0]=idx 0, A[0,1]=idx 1, A[1,0]=idx 2, A[1,1]=idx 3
        # B is 2x2: B[0,0]=idx 0, B[0,1]=idx 1, B[1,0]=idx 2, B[1,1]=idx 3 (relative to k_offset)
        # C is 2x2: C[0,0]=idx 0, C[0,1]=idx 1, C[1,0]=idx 2, C[1,1]=idx 3 (relative to k_offset)
        
        # U indices (A): 0,1,2,3
        # V indices (B): 0,1,2,3 (mapped to j*p + k)
        # W indices (C): 0,1,2,3 (mapped to k*n + i)
        
        # We need to map local indices to global indices.
        # Local A idx: i*m + j
        # Local B idx: j*p + (k_offset + j_local)
        # Local C idx: (k_offset + k_local)*n + i
        
        terms = []
        
        # M1: A[0,0] * (B[0,1] - B[1,1])
        # U: A[0,0] -> 1
        # V: B[0,1] -> 1, B[1,1] -> -1
        # W: C[0,1] -> 1, C[1,1] -> 1
        terms.append({
            'u': {0: 1},
            'v': {1: 1, 3: -1},
            'w': {1: 1, 3: 1}
        })
        
        # M2: (A[0,0] + A[1,0]) * B[1,1]
        terms.append({
            'u': {0: 1, 2: 1},
            'v': {3: 1},
            'w': {2: 1, 3: 1}
        })
        
        # M3: (A[1,1] - A[0,1]) * B[0,0]
        terms.append({
            'u': {3: 1, 1: -1},
            'v': {0: 1},
            'w': {0: 1, 2: 1}
        })
        
        # M4: A[1,1] * (B[0,0] + B[1,0])
        terms.append({
            'u': {3: 1},
            'v': {0: 1, 2: 1},
            'w': {0: -1, 2: -1}
        })
        
        # M5: (A[0,0] + A[1,1]) * (B[1,0] + B[1,1])
        terms.append({
            'u': {0: 1, 3: 1},
            'v': {2: 1, 3: 1},
            'w': {0: 1, 1: -1}
        })
        
        # M6: (A[0,1] - A[1,1]) * (B[0,0] + B[1,0])
        terms.append({
            'u': {1: 1, 3: -1},
            'v': {0: 1, 2: 1},
            'w': {1: 1, 3: 1}
        })
        
        # M7: (A[0,0] - A[0,1]) * (B[0,1] + B[1,1])
        terms.append({
            'u': {0: 1, 1: -1},
            'v': {1: 1, 3: 1},
            'w': {0: 1, 1: 1}
        })
        
        for t in terms:
            c = start_col
            for u_idx, u_val in t['u'].items():
                U[u_idx, c] = u_val
            for v_idx, v_val in t['v'].items():
                # v_idx is local B index (j*p + k_local)
                # But our V matrix is indexed by j*p + k_global
                # In the block, k_global = k_offset + v_local_k
                # v_idx in our dict is just the local linear index of B (0..3)
                # B local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # Global B index: j*p + (k_offset + k_local)
                # If v_idx=0: j=0, k_local=0 -> j*p + k_offset
                # If v_idx=1: j=0, k_local=1 -> j*p + k_offset + 1
                # If v_idx=2: j=1, k_local=0 -> j*p + k_offset
                # If v_idx=3: j=1, k_local=1 -> j*p + k_offset + 1
                
                # We need to map v_idx to (j, k_local)
                j_local = v_idx // 2
                k_local = v_idx % 2
                global_v_idx = j_local * p + (k_offset + k_local)
                V[global_v_idx, c] = v_val
                
            for w_idx, w_val in t['w'].items():
                # w_idx is local C index (k_local*n + i)
                # C local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # k_local = w_idx // n
                # i_local = w_idx % n
                k_local = w_idx // n
                i_local = w_idx % n
                global_w_idx = (k_offset + k_local) * n + i_local
                W[global_w_idx, c] = w_val
                
            start_col += 1
            
        return start_col

    def add_schoolbook_block(U, V, W, start_col, k_offset):
        """
        Add a 2x2x1 schoolbook block for column k_offset.
        Returns the number of columns used (4).
        """
        for i in range(n):
            for j in range(m):
                c = start_col
                U[i * m + j, c] = 1
                V[j * p + (k_offset), c] = 1
                W[(k_offset) * n + i, c] = 1
                start_col += 1
        return start_col

    # Construct the block decomposition
    # Number of 2x2 blocks
    num_2x2_blocks = p // 2
    has_1x1_block = (p % 2 == 1)
    
    total_rank = num_2x2_blocks * 7 + (4 if has_1x1_block else 0)
    
    if total_rank < best_rank:
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        # Add 2x2 blocks
        for b in range(num_2x2_blocks):
            k_offset = b * 2
            col = add_strassen_block(U, V, W, col, k_offset)
            
        # Add 1x1 block if needed
        if has_1x1_block:
            k_offset = num_2x2_blocks * 2
            col = add_schoolbook_block(U, V, W, col, k_offset)
            
        if verify_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return best_result
# EVOLVE-BLOCK-END
