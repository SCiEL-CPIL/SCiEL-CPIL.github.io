import numpy as np
import time

def construct():
    """
    Returns a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses block-wise Strassen algorithm.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = 2, 2, 14
    
    # Number of 2x2 blocks in the 2x14 matrix B
    num_blocks = p // 2
    # Rank of 2x2 matrix multiplication (Strassen)
    strassen_rank = 7
    total_rank = num_blocks * strassen_rank
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 49)
    # V: (m*p, rank) = (28, 49)
    # W: (p*n, rank) = (28, 49)
    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)
    
    # Strassen coefficients for 2x2 * 2x2
    # We define the 7 products P1..P7
    # Each product corresponds to a rank-1 term in the decomposition.
    
    for t in range(num_blocks):
        k_offset = t * 2
        
        for s in range(strassen_rank):
            col = t * strassen_rank + s
            
            if s == 0:
                # P1 = (A00 + A11)(B00 + B11)
                # x = A00 + A11
                # y = B00 + B11
                # C00 += P1, C11 += P1
                U[0, col] += 1   # A00
                U[3, col] += 1   # A11
                V[0 * p + k_offset + 0, col] += 1   # B00
                V[1 * p + k_offset + 1, col] += 1   # B11
                W[(k_offset + 0) * n + 0, col] += 1 # C00
                W[(k_offset + 1) * n + 1, col] += 1 # C11
                
            elif s == 1:
                # P2 = (A10 + A11) B00
                # x = A10 + A11
                # y = B00
                # C10 += P2, C11 -= P2
                U[2, col] += 1   # A10
                U[3, col] += 1   # A11
                V[0 * p + k_offset + 0, col] += 1   # B00
                W[(k_offset + 0) * n + 1, col] += 1 # C10
                W[(k_offset + 1) * n + 1, col] += -1 # C11
                
            elif s == 2:
                # P3 = A00 (B01 - B11)
                # x = A00
                # y = B01 - B11
                # C01 += P3, C11 += P3
                U[0, col] += 1   # A00
                V[0 * p + k_offset + 1, col] += 1   # B01
                V[1 * p + k_offset + 1, col] += -1  # B11
                W[(k_offset + 1) * n + 0, col] += 1 # C01
                W[(k_offset + 1) * n + 1, col] += 1 # C11
                
            elif s == 3:
                # P4 = A11 (B10 - B00)
                # x = A11
                # y = B10 - B00
                # C00 += P4, C10 += P4
                U[3, col] += 1   # A11
                V[1 * p + k_offset + 0, col] += 1   # B10
                V[0 * p + k_offset + 0, col] += -1  # B00
                W[(k_offset + 0) * n + 0, col] += 1 # C00
                W[(k_offset + 0) * n + 1, col] += 1 # C10
                
            elif s == 4:
                # P5 = (A00 + A01) B11
                # x = A00 + A01
                # y = B11
                # C00 -= P5, C01 += P5
                U[0, col] += 1   # A00
                U[1, col] += 1   # A01
                V[1 * p + k_offset + 1, col] += 1   # B11
                W[(k_offset + 0) * n + 0, col] += -1 # C00
                W[(k_offset + 1) * n + 0, col] += 1 # C01
                
            elif s == 5:
                # P6 = (A10 - A00) (B00 + B01)
                # x = A10 - A00
                # y = B00 + B01
                # C11 += P6
                U[0, col] += -1  # A00
                U[2, col] += 1   # A10
                V[0 * p + k_offset + 0, col] += 1   # B00
                V[0 * p + k_offset + 1, col] += 1   # B01
                W[(k_offset + 1) * n + 1, col] += 1 # C11
                
            elif s == 6:
                # P7 = (A01 - A11) (B10 + B11)
                # x = A01 - A11
                # y = B10 + B11
                # C00 += P7
                U[1, col] += 1   # A01
                U[3, col] += -1  # A11
                V[1 * p + k_offset + 0, col] += 1   # B10
                V[1 * p + k_offset + 1, col] += 1   # B11
                W[(k_offset + 0) * n + 0, col] += 1 # C00

    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Quick verification using tensor contraction
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    if time.time() < deadline:
        # Reshape for efficient contraction
        U_r = np.array(U_list).reshape(n, m, total_rank)
        V_r = np.array(V_list).reshape(m, p, total_rank)
        W_r = np.array(W_list).reshape(p, n, total_rank)
        
        # Compute T[i, j, k]
        # T[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
        # This is a batched contraction.
        # We can do this by summing over the last axis of the product of the three tensors.
        # But they have different shapes.
        # Let's just loop over i, j, k for verification since n, m, p are small.
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    val = np.dot(u_val, v_val * w_val)
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
        
        if max_err > 1e-9:
            # Fallback to schoolbook if verification fails
            rank_fb = n * m * p
            U_fb = np.zeros((n * m, rank_fb), dtype=np.int8)
            V_fb = np.zeros((m * p, rank_fb), dtype=np.int8)
            W_fb = np.zeros((p * n, rank_fb), dtype=np.int8)
            col = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_fb[ii * m + jj, col] = 1
                        V_fb[jj * p + kk, col] = 1
                        W_fb[kk * n + ii, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
            
    return {"U": U_list, "V": V_list, "W": W_list}
