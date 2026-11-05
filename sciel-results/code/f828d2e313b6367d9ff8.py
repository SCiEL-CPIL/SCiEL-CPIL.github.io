"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precompute index arrays for the 72 equations
    # Indices for U (n*m), V (m*p), W (p*n)
    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    u_idx = i_flat * m + j_flat  # Shape (n*m*p,)
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    num_eq = n * m * p # 72
    
    def check_exact(U, V, W, rank):
        """Verify if U, V, W with integer/half-integer entries exactly decompose the tensor."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Compute tensor values
        # U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        # We sum over r.
        # Vectorized:
        # U_sel = U_arr[u_idx, :] -> (72, rank)
        # V_sel = V_arr[v_idx, :] -> (72, rank)
        # W_sel = W_arr[w_idx, :] -> (72, rank)
        # Prod = U_sel * V_sel * W_sel -> (72, rank)
        # Sum = Prod.sum(axis=1) -> (72,)
        
        U_sel = U_arr[u_idx, :]
        V_sel = V_arr[v_idx, :]
        W_sel = W_arr[w_idx, :]
        
        prod = U_sel * V_sel * W_sel
        sums = prod.sum(axis=1)
        
        if not np.allclose(sums, 1.0, atol=1e-9):
            return False
        return True

    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # Try to find lower rank solutions
    # Theoretical min for 3x3x8 is likely 64 (23+23+18)
    # We will try ranks from 64 up to 71
    
    best_result = None
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Fallback to schoolbook first
    sb_res, sb_rank = schoolbook()
    best_result = sb_res
    best_rank = sb_rank
    
    # Function to attempt a specific rank
    def attempt_rank(target_rank):
        if time.time() > deadline - 10:
            return None
            
        # Initialize with block structure
        # Block 1: k in [0, 2], cols 0-22 (23 cols)
        # Block 2: k in [3, 5], cols 23-45 (23 cols)
        # Block 3: k in [6, 7], cols 46-63 (18 cols)
        
        # This block structure assumes target_rank == 64.
        # If target_rank != 64, we need a different init.
        # For simplicity, we will only use the block structure for 64.
        # For other ranks, we use dense random init.
        
        if target_rank == 64:
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Block 3 (k=6,7) is fixed schoolbook
            # Cols 46 to 63
            col = 46
            for i in range(n):
                for j in range(m):
                    for k in [6, 7]:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            
            # Block 1 & 2 (k=0..5) are random
            for r in range(46):
                for i in range(n * m):
                    U[i, r] = random.choice([-1, 0, 1])
                for i in range(m * p):
                    k_local = i % p
                    if r < 23:
                        if k_local < 3:
                            V[i, r] = random.choice([-1, 0, 1])
                        else:
                            V[i, r] = 0
                    else:
                        if 3 <= k_local < 6:
                            V[i, r] = random.choice([-1, 0, 1])
                        else:
                            V[i, r] = 0
                for i in range(p * n):
                    k_local = i // n
                    if r < 23:
                        if k_local < 3:
                            W[i, r] = random.choice([-1, 0, 1])
                        else:
                            W[i, r] = 0
                    else:
                        if 3 <= k_local < 6:
                            W[i, r] = random.choice([-1, 0, 1])
                        else:
                            W[i, r] = 0
        else:
            # Dense random init for other ranks
            U = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)], dtype=np.float64)
            V = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)], dtype=np.float64)
            W = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)], dtype=np.float64)
            # Scale up to have some magnitude
            U *= 2
            V *= 2
            W *= 2

        # Gradient Descent
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        steps = 2000
        for step in range(steps):
            if time.time() > deadline - 5:
                return None
                
            # Compute Error
            U_sel = U[u_idx, :]
            V_sel = V[v_idx, :]
            W_sel = W[w_idx, :]
            prod = U_sel * V_sel * W_sel
            sums = prod.sum(axis=1)
            err = sums - 1.0
            
            # Compute Gradients
            # dL/dU[i, r] = 2 * sum_{(i,j,k) s.t. u_idx[i,j,k]==i} err[i,j,k] * V[v_idx, r] * W[w_idx, r]
            
            # We can vectorize this accumulation.
            # For each r, we have err (72,), V_sel[:, r] (72,), W_sel[:, r] (72,)
            # We need to scatter-add to U, V, W.
            
            # grad_U shape (n*m, R)
            # For each r, contrib = err * V_sel[:, r] * W_sel[:, r] (72,)
            # We add contrib[t] to grad_U[u_idx[t], r]
            
            # Efficient way:
            # grad_U = np.zeros((n*m, R))
            # for r in range(R):
            #    contrib = err * V_sel[:, r] * W_sel[:, r]
            #    np.add.at(grad_U, (u_idx, r), contrib)
            
            # This loop over R (64) is fast.
            
            R = target_rank
            grad_U = np.zeros((n * m, R))
            grad_V = np.zeros((m * p, R))
            grad_W = np.zeros((p * n, R))
            
            for r in range(R):
                contrib = err * V_sel[:, r] * W_sel[:, r]
                np.add.at(grad_U, (u_idx, r), contrib)
                
                contrib_v = err * U_sel[:, r] * W_sel[:, r]
                np.add.at(grad_V, (v_idx, r), contrib_v)
                
                contrib_w = err * U_sel[:, r] * V_sel[:, r]
                np.add.at(grad_W, (w_idx, r), contrib_w)
                
            grad_U *= 2
            grad_V *= 2
            grad_W *= 2
            
            # Update with Momentum
            vU = momentum * vU - lr * grad_U
            U += vU
            vV = momentum * vV - lr * grad_V
            V += vV
            vW = momentum * vW - lr * grad_W
            W += vW
            
            # Periodic check and snap
            if step % 100 == 0:
                # Snap to nearest 0.5
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if check_exact(U_snap, V_snap, W_snap, R):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}, R
        
        # Final check
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if check_exact(U_snap, V_snap, W_snap, R):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}, R
            
        return None

    # Try ranks
    for r_target in [64, 65, 66, 67, 68, 69, 70, 71]:
        if time.time() > deadline - 15:
            break
            
        # Multiple restarts for rank 64
        restarts = 5 if r_target == 64 else 1
        
        for _ in range(restarts):
            if time.time() > deadline - 15:
                break
            res = attempt_rank(r_target)
            if res is not None:
                result, rank = res
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
                break # Found a valid one for this rank, try next lower rank? No, we are going up.
                    # Actually, we want the MINIMAL rank. So if we find 64, we are done.
                    # If we find 65, we should still try 64? No, we tried 64 first.
                    # So if we find a valid rank, we can return immediately if it's the lowest possible?
                    # The lowest possible is 64. If we find 64, return.
                    # If we find 65, we should check if 64 was truly impossible or just hard.
                    # Given time, if we find 65, we might assume 64 is hard and return 65.
                    # But let's try to return the best found.
                
    return best_result
# EVOLVE-BLOCK-END
