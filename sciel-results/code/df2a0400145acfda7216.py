"""Matrix multiplication 2x2 by 2x8 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit
    
    n, m, p = 2, 2, 8
    
    # The matrix multiplication tensor T has indices (i, j, k) where:
    # i in 0..n-1 (row of A / row of C)
    # j in 0..m-1 (col of A / row of B)
    # k in 0..p-1 (col of B / col of C)
    # T[i,j,k] = 1 if this is a valid multiplication term, else 0
    # Actually T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    
    # The tensor decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 for all i,j,k (since every combination contributes to the product)
    # Actually no. The multiplication tensor M_{n,m,p} has:
    # M_{i,j,k} = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Since C[i,k] = sum_j A[i,j] * B[j,k], every (i,j,k) triple has coefficient 1.
    
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is equivalent to: the tensor T with T[i,j,k] = 1 for all i,j,k
    # can be decomposed as sum_r u_r ⊗ v_r ⊗ w_r where:
    # u_r[i*m+j] = U[i*m+j, r]
    # v_r[j*p+k] = V[j*p+k, r]  
    # w_r[k*n+i] = W[k*n+i, r]
    
    # Strategy: Use Strassen's algorithm for each 2x2 block of B.
    # B is 2x8, so we can split it into 4 blocks of 2x2.
    # For each block, use rank-7 Strassen decomposition.
    # Total rank = 4 * 7 = 28.
    
    # But we can do better by exploiting the shared structure.
    # Actually, let's first verify that rank 28 works with correct Strassen.
    
    # Strassen for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # In our tensor notation for a single 2x2 block:
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0,1} (local)
    # C indices: i in {0,1}, k in {0,1} (local)
    
    # For the full problem with 4 blocks:
    # Block b (b=0,1,2,3): B's columns are 2b, 2b+1
    # A is shared across all blocks
    # C's columns for block b are 2b, 2b+1
    
    # Let me build the rank-28 decomposition:
    rank = 28
    U = np.zeros((n * m, rank))  # (4, 28)
    V = np.zeros((m * p, rank))  # (16, 28)
    W = np.zeros((p * n, rank))  # (16, 28)
    
    for b in range(4):
        base = b * 7
        # Local indices: B row 0, col 2b -> V index 0*8+2b = 2b
        # B row 0, col 2b+1 -> V index 0*8+2b+1 = 2b+1
        # B row 1, col 2b -> V index 1*8+2b = 8+2b
        # B row 1, col 2b+1 -> V index 1*8+2b+1 = 8+2b+1
        
        # C row 0, col 2b -> W index 2b*2+0 = 4b
        # C row 0, col 2b+1 -> W index (2b+1)*2+0 = 4b+2
        # C row 1, col 2b -> W index 2b*2+1 = 4b+1
        # C row 1, col 2b+1 -> W index (2b+1)*2+1 = 4b+3
        
        # A[0,0] -> U index 0*2+0 = 0
        # A[0,1] -> U index 0*2+1 = 1
        # A[1,0] -> U index 1*2+0 = 2
        # A[1,1] -> U index 1*2+1 = 3
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U[0, base+0] += 1
        U[3, base+0] += 1
        V[2*b, base+0] += 1
        V[8+2*b+1, base+0] += 1
        W[4*b+0, base+0] += 1   # C[0,2b]
        W[4*b+3, base+0] += 1   # C[1,2b+1]
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U[2, base+1] += 1
        U[3, base+1] += 1
        V[2*b, base+1] += 1
        W[4*b+1, base+1] += 1   # C[1,2b]
        W[4*b+3, base+1] -= 1   # C[1,2b+1]
        
        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U[0, base+2] += 1
        V[2*b+1, base+2] += 1
        V[8+2*b+1, base+2] -= 1
        W[4*b+2, base+2] += 1   # C[0,2b+1]
        W[4*b+3, base+2] += 1   # C[1,2b+1]
        
        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U[3, base+3] += 1
        V[8+2*b, base+3] += 1
        V[2*b, base+3] -= 1
        W[4*b+0, base+3] += 1   # C[0,2b]
        W[4*b+1, base+3] += 1   # C[1,2b]
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U[0, base+4] += 1
        U[1, base+4] += 1
        V[8+2*b+1, base+4] += 1
        W[4*b+0, base+4] -= 1   # C[0,2b]
        W[4*b+2, base+4] += 1   # C[0,2b+1]
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U[2, base+5] += 1
        U[0, base+5] -= 1
        V[2*b, base+5] += 1
        V[2*b+1, base+5] += 1
        W[4*b+3, base+5] += 1   # C[1,2b+1]
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U[1, base+6] += 1
        U[3, base+6] -= 1
        V[8+2*b, base+6] += 1
        V[8+2*b+1, base+6] += 1
        W[4*b+0, base+6] += 1   # C[0,2b]
    
    # Verify correctness
    A_test = np.array([[1.0, 2.0], [3.0, 4.0]])
    B_test = np.array([[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]])
    C_test = A_test @ B_test
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for r in range(rank):
                for j in range(m):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_decomp[i, k] = total
    
    if not np.allclose(C_decomp, C_test, atol=1e-10):
        # Fallback to schoolbook
        rank_s = n * m * p
        U_s = np.zeros((n*m, rank_s))
        V_s = np.zeros((m*p, rank_s))
        W_s = np.zeros((p*n, rank_s))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U_s[i*m+j, r] = 1
                    V_s[j*p+k, r] = 1
                    W_s[k*n+i, r] = 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Check if we can find a lower rank via optimization
    # Try to see if rank < 28 is possible
    # For now, return the rank-28 solution
    
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt to find a lower-rank decomposition using numerical optimization
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want min rank such that T = sum_r u_r ⊗ v_r ⊗ w_r
    # with u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    
    # This is a hard NP-hard problem in general, but for small tensors we can try
    # Let's try rank 24, 25, 26, 27 using gradient-based optimization
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    for target_rank in [27, 26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline - 20:
            break
        
        # Initialize randomly
        np.random.seed(target_rank * 1000 + 42)
        
        success = False
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            U_r = np.random.randn(n*m, target_rank) * 0.5
            V_r = np.random.randn(m*p, target_rank) * 0.5
            W_r = np.random.randn(p*n, target_rank) * 0.5
            
            # Simple gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 20:
                    break
                
                # Compute current tensor estimate
                # T_est[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                T_est = np.zeros((n, m, p))
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_est[i,j,k] += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                
                # Target: all ones
                T_target = np.ones((n, m, p))
                error = np.sum((T_est - T_target)**2)
                
                if error < 1e-8:
                    success = True
                    break
                
                # Compute gradients
                dU = np.zeros_like(U_r)
                dV = np.zeros_like(V_r)
                dW = np.zeros_like(W_r)
                
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                diff = T_est[i,j,k] - T_target[i,j,k]
                                dU[i*m+j, r] += diff * V_r[j*p+k, r] * W_r[k*n+i, r]
                                dV[j*p+k, r] += diff * U_r[i*m+j, r] * W_r[k*n+i, r]
                                dW[k*n+i, r] += diff * U_r[i*m+j, r] * V_r[j*p+k, r]
                
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
                
                # Adaptive learning rate
                if step % 50 == 0 and step > 0:
                    lr *= 0.95
            
            if success:
                break
        
        if success:
            best_rank = target_rank
            best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            break
    
    return best_result
# EVOLVE-BLOCK-END
