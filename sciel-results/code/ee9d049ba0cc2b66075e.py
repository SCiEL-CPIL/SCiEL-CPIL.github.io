"""Rewritten program for matrix multiplication tensor decomposition 4x4x7."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def make_schoolbook():
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

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (a00+a11)*(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2: (a10+a11)*b00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3: a00*(b01-b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4: a11*(b10-b00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[0, 3] = 1
        
        # M5: (a00+a01)*b11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6: (a10-a00)*(b00+b01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[3, 5] = 1
        
        # M7: (a01-a11)*(b10+b11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    def schoolbook_2x2x1():
        """Return (U, V, W) for 2x2x1 schoolbook decomposition, rank 2."""
        U = np.zeros((4, 2))
        V = np.zeros((2, 2))
        W = np.zeros((2, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[2, 1] = 1; V[0, 1] = 1; W[1, 1] = 1
        
        return U, V, W

    def schoolbook_4x4x1():
        """Return (U, V, W) for 4x4x1 schoolbook decomposition, rank 4."""
        U = np.zeros((16, 4))
        V = np.zeros((4, 4))
        W = np.zeros((4, 4))
        for j in range(4):
            for i in range(4):
                U[i * 4 + j, j] = 1
            V[j, j] = 1
            W[i, i] = 1
        return U, V, W

    def kronecker_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
        """
        Combine two decompositions via Kronecker product.
        First: n1 x m1 x p1 with rank r1
        Second: n2 x m2 x p2 with rank r2
        Result: (n1*n2) x (m1*m2) x (p1*p2) with rank r1*r2
        """
        r1 = W1.shape[1]
        r2 = W2.shape[1]
        r = r1 * r2
        
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1 * r2 + c2
                
                # U combination
                for i1 in range(n1):
                    for j1 in range(m1):
                        val1 = U1[i1 * m1 + j1, c1]
                        if val1 == 0:
                            continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                val2 = U2[i2 * m2 + j2, c2]
                                if val2 == 0:
                                    continue
                                row = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                                U[row, c] += val1 * val2
                
                # V combination
                for j1 in range(m1):
                    for k1 in range(p1):
                        val1 = V1[j1 * p1 + k1, c1]
                        if val1 == 0:
                            continue
                        for j2 in range(m2):
                            for k2 in range(p2):
                                val2 = V2[j2 * p2 + k2, c2]
                                if val2 == 0:
                                    continue
                                row = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                                V[row, c] += val1 * val2
                
                # W combination
                for k1 in range(p1):
                    for i1 in range(n1):
                        val1 = W1[k1 * n1 + i1, c1]
                        if val1 == 0:
                            continue
                        for k2 in range(p2):
                            for i2 in range(n2):
                                val2 = W2[k2 * n2 + i2, c2]
                                if val2 == 0:
                                    continue
                                row = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                                W[row, c] += val1 * val2
        
        return U, V, W

    def construct_block_decomp():
        """
        Construct a decomposition for 4x4x7 using block structure.
        Split the 7 output columns into groups: [2, 2, 2, 1]
        Use Strassen-based 4x4x2 (rank 14) for first three groups.
        Use schoolbook 4x4x1 (rank 4) for last group.
        Total rank: 3*14 + 4 = 46
        """
        n, m, p = 4, 4, 7
        
        U_s, V_s, W_s = strassen_2x2x2()
        U_sb, V_sb, W_sb = schoolbook_2x2x1()
        
        # Kronecker product: 4x4x2 with rank 14
        U_442, V_442, W_442 = kronecker_decomp(U_s, V_s, W_s, U_sb, V_sb, W_sb, 2, 2, 2, 2, 2, 1)
        
        # 4x4x1 schoolbook
        U_441, V_441, W_441 = schoolbook_4x4x1()
        
        total_rank = 3 * 14 + 4
        U = np.zeros((16, total_rank))
        V = np.zeros((4*7, total_rank))
        W = np.zeros((7*4, total_rank))
        
        col_offset = 0
        
        # Copy 1: columns 0-1 of B
        U[:, col_offset:col_offset+14] = U_442
        V[0:8, col_offset:col_offset+14] = V_442
        W[0:8, col_offset:col_offset+14] = W_442
        col_offset += 14
        
        # Copy 2: columns 2-3 of B
        U[:, col_offset:col_offset+14] = U_442
        V[8:16, col_offset:col_offset+14] = V_442
        W[8:16, col_offset:col_offset+14] = W_442
        col_offset += 14
        
        # Copy 3: columns 4-5 of B
        U[:, col_offset:col_offset+14] = U_442
        V[16:24, col_offset:col_offset+14] = V_442
        W[16:24, col_offset:col_offset+14] = W_442
        col_offset += 14
        
        # Copy 4: column 6 of B
        U[:, col_offset:col_offset+4] = U_441
        V[24:28, col_offset:col_offset+4] = V_441
        W[24:28, col_offset:col_offset+4] = W_441
        col_offset += 4
        
        return U, V, W, total_rank

    # Try the block decomposition
    try:
        U_block, V_block, W_block, rank_block = construct_block_decomp()
        err = verify(U_block, V_block, W_block, rank_block)
        if err < 1e-8:
            best_rank = rank_block
            best_result = {
                "U": U_block.tolist(),
                "V": V_block.tolist(),
                "W": W_block.tolist()
            }
    except Exception as e:
        pass

    # If we have the block decomposition, try to improve with optimization
    if best_rank < n * m * p:
        # Try to find a lower-rank decomposition using optimization
        # Start from the block decomposition and try to reduce rank
        
        U_init = np.array(best_result["U"], dtype=float)
        V_init = np.array(best_result["V"], dtype=float)
        W_init = np.array(best_result["W"], dtype=float)
        
        # Try various target ranks below the current best
        for target_rank in range(30, best_rank):
            if time.time() > deadline - 15:
                break
            
            found = False
            for trial in range(5):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(trial * 104729 + target_rank * 31)
                
                # Initialize with random perturbation of a subset of columns
                # Or try a fresh random initialization
                if trial == 0:
                    # Use a subset of the existing decomposition
                    if target_rank <= U_init.shape[1]:
                        U = U_init[:, :target_rank].copy()
                        V = V_init[:, :target_rank].copy()
                        W = W_init[:, :target_rank].copy()
                    else:
                        U = np.random.randn(n * m, target_rank) * 0.3
                        V = np.random.randn(m * p, target_rank) * 0.3
                        W = np.random.randn(p * n, target_rank) * 0.3
                else:
                    U = np.random.randn(n * m, target_rank) * 0.3
                    V = np.random.randn(m * p, target_rank) * 0.3
                    W = np.random.randn(p * n, target_rank) * 0.3
                
                # Gradient descent with adaptive learning rate
                for it in range(200):
                    if time.time() > deadline - 5:
                        break
                    
                    u_vals = U[u_idx, :]
                    v_vals = V[v_idx, :]
                    w_vals = W[w_idx, :]
                    
                    prod = u_vals * v_vals * w_vals
                    result = prod.sum(axis=1)
                    error = result - 1.0
                    err_norm = np.linalg.norm(error)
                    
                    if err_norm < 1e-10:
                        break
                    
                    # Compute gradients
                    # grad_U[i*m+j, c] = sum_{k} error[i*m*p+j*p+k] * V[j*p+k, c] * W[k*n+i, c]
                    # grad_V[j*p+k, c] = sum_{i} error[i*m*p+j*p+k] * U[i*m+j, c] * W[k*n+i, c]
                    # grad_W[k*n+i, c] = sum_{j} error[i*m*p+j*p+k] * U[i*m+j, c] * V[j*p+k, c]
                    
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            u_row = U[i * m + j, :]
                            for k in range(p):
                                e = error[i * m * p + j * p + k]
                                if abs(e) > 1e-12:
                                    v_row = V[j * p + k, :]
                                    w_row = W[k * n + i, :]
                                    grad_U[i * m + j, :] += e * v_row * w_row
                                    grad_V[j * p + k, :] += e * u_row * w_row
                                    grad_W[k * n + i, :] += e * u_row * v_row
                    
                    # Adaptive learning rate
                    grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                    if grad_norm < 1e-15:
                        break
                    
                    lr = min(0.5, 1.0 / (1.0 + grad_norm))
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                
                err = verify(U, V, W, target_rank)
                if err < 1e-8:
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    found = True
                    break
            
            if found:
                # Continue trying lower ranks
                continue
            else:
                # If we couldn't find a solution at this rank, try a few lower ones
                # but break if we've tried several
                pass

    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
