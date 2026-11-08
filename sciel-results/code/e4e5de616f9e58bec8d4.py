import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Target rank to try
    target_rank = 38
    
    # Build the target tensor T
    # T[i, j, k] is 1 if A[i,j] B[j,k] contributes to C[i,k]
    # In the standard multiplication, it always does.
    # So T[i, j, k] = 1 for all i, j, k.
    # Wait, the tensor is in the space of (A, B, C).
    # The decomposition is T = sum u_r v_r w_r.
    # The indices of u_r are (i, j), v_r are (j, k), w_r are (k, i).
    # So u_r is a vector of length n*m.
    # v_r is a vector of length m*p.
    # w_r is a vector of length p*n.
    
    # The condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k.
    
    # Let's verify this condition for the rank 39 solution.
    # If this condition is correct, then the tensor is indeed the "all-ones" tensor in the sense of the values,
    # but the rank is determined by the structure.
    
    # We will try to find a rank 38 solution using gradient descent.
    
    def try_find_decomposition(rank, time_limit=10):
        np.random.seed(42)
        # Initialize
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Precompute indices for efficiency
        i_vals = np.arange(n)
        j_vals = np.arange(m)
        k_vals = np.arange(p)
        
        start_time = time.time()
        lr = 0.01
        step = 0
        
        while time.time() - start_time < time_limit:
            # Compute current tensor approximation
            # T_approx[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We want T_approx == 1 for all i, j, k.
            
            # Calculate error
            error = 0.0
            # Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Calculate sum for this (i, j, k)
                        s = 0.0
                        # We need to accumulate gradients for each r
                        # But we can do it in a vectorized way per (i,j,k)
                        
                        # U_idx = i*m + j
                        # V_idx = j*p + k
                        # W_idx = k*n + i
                        
                        u_val = U[i*m + j, :]
                        v_val = V[j*p + k, :]
                        w_val = W[k*n + i, :]
                        
                        # s = dot(u_val, v_val * w_val)
                        s = np.dot(u_val, v_val * w_val)
                        err = 1.0 - s
                        error += err * err
                        
                        # Gradients
                        # d/dU: err * (v_val * w_val)
                        # d/dV: err * (u_val * w_val)
                        # d/dW: err * (u_val * v_val)
                        
                        dU[i*m + j, :] += err * (v_val * w_val)
                        dV[j*p + k, :] += err * (u_val * w_val)
                        dW[k*n + i, :] += err * (u_val * v_val)
            
            if error < 1e-10:
                return U, V, W
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if step % 100 == 0:
                lr *= 0.95
            step += 1
        
        return None

    # Try to find a rank 38 solution
    result = try_find_decomposition(38, time_limit=20)
    
    if result:
        U, V, W = result
        # Convert to lists of integers/half-integers if possible
        # The problem allows integer/half-integer entries.
        # We can round to nearest 0.5
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify the rounded solution
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_int[i*m + j, :], V_int[j*p + k, :] * W_int[k*n + i, :])
                    if abs(s - 1.0) > 1e-9:
                        err += abs(s - 1.0)
        
        if err < 1e-6:
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fallback to rank 39 solution
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    def add_strassen_block(k_start):
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        add_term([0, 4], [b00, b11], [c00, c11])
        add_term([3, 4], [b00], [c10, c11])
        W[c11][col-1] = -1
        
        add_term([0], [b01, b11], [c01, c11])
        V[b11][col-1] = -1
        
        add_term([4], [b10, b00], [c00, c10])
        V[b00][col-1] = -1
        
        add_term([0, 1], [b11], [c00, c01])
        W[c00][col-1] = -1
        
        add_term([3, 0], [b00, b01], [c11])
        U[0][col-1] = -1
        
        add_term([1, 4], [b10, b11], [c00])
        U[4][col-1] = -1

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
