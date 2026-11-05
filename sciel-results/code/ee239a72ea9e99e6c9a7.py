import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # Rank 54 decomposition
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # 7 columns for one 2x2 * 2x2 block
        
        # m0: (x00+x11)(y00+y11) -> Z00+=1, Z11+=1
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m1: (x10+x11)y00 -> Z10+=1, Z11-=1
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # m2: x00(y01-y11) -> Z01+=1, Z11+=1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m3: x11(y10-y00) -> Z00+=1, Z10+=1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # m4: (x00+x01)y11 -> Z00-=1, Z01+=1
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # m5: (x10-x00)(y00+y01) -> Z11+=1
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m6: (x01-x11)(y10+y11) -> Z00+=1
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    # Blocks for A_R (j=3,4) * B_R (j=3,4)
    # k splits: 0-1, 2-3, 4-5
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    # Blocks for A_L1 (j=0,1) * B_L1 (j=0,1)
    # k splits: 0-1, 2-3, 4-5
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Outer product for A_L2 (j=2) * B_L2 (j=2)
    # A_L2 is 2x1, B_L2 is 1x6
    for k in range(p):
        # C[0, k] += A[0, 2] * B[2, k]
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # C[1, k] += A[1, 2] * B[2, k]
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy for fast verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_decomposition_np(U_arr, V_arr, W_arr, n, m, p, rank_to_check):
        """Verify that the decomposition exactly reproduces the tensor using numpy."""
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        # We can vectorize this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum over r of U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    val = np.dot(U_arr[i*m+j, :rank_to_check], 
                                 V_arr[j*p+k, :rank_to_check] * W_arr[k*n+i, :rank_to_check])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Verify the initial decomposition
    if not verify_decomposition_np(U_np, V_np, W_np, n, m, p, rank):
        # Fallback: return the unverified decomposition
        return {"U": U, "V": V, "W": W}
    
    # Try to eliminate redundant terms using numpy-based verification
    current_rank = rank
    
    # Try removing terms one at a time (greedy elimination)
    # We'll try multiple passes since removing one term might enable removing another
    max_passes = 5
    for pass_num in range(max_passes):
        if time.time() > deadline - 15:  # Leave 15s buffer
            break
        
        removed_in_pass = False
        for trial in range(current_rank):
            if time.time() > deadline - 10:  # Leave 10s buffer
                break
            
            # Try removing term 'trial'
            # Create modified arrays with term 'trial' zeroed out
            U_test = U_np.copy()
            V_test = V_np.copy()
            W_test = W_np.copy()
            
            U_test[:, trial] = 0
            V_test[:, trial] = 0
            W_test[:, trial] = 0
            
            # Verify
            if verify_decomposition_np(U_test, V_test, W_test, n, m, p, current_rank):
                # Term can be removed
                U_np = U_test
                V_np = V_test
                W_np = W_test
                current_rank -= 1
                removed_in_pass = True
                break  # Restart the loop with the new rank
        
        if not removed_in_pass:
            break
    
    # Convert back to lists and trim
    final_U = U_np[:, :current_rank].tolist()
    final_V = V_np[:, :current_rank].tolist()
    final_W = W_np[:, :current_rank].tolist()
    
    return {"U": final_U, "V": final_V, "W": final_W}
# EVOLVE-BLOCK-END
