"""Program to find a rank-minimal tensor decomposition for 3x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE = 145.0  # 150s limit, use 145s for safety

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strassen 2x2x2 decomposition
    # Local indices 0..3 for 2x2 matrices
    # u, v, w are length 4 vectors
    strassen_u = [
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11+A12
        [-1, 0, 1, 0],# M6: A21-A11
        [0, 1, 0, -1] # M7: A12-A22
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12-B22
        [0, 0, 1, -1],# M4: B21-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ]
    strassen_w = [
        [1, 0, 0, 1], # C11=1, C22=1
        [0, 0, 1, -1],# C21=1, C22=-1
        [0, 1, 0, 1], # C12=1, C22=1
        [1, 0, 1, 0], # C11=1, C21=1
        [-1, 1, 0, 0],# C11=-1, C12=1
        [0, 0, 0, 1], # C22=1
        [1, 0, 0, 1]  # C11=1, C22=1
    ]
    
    # Define blocks: (i_start, j_start, k_start)
    blocks = [
        (0, 0, 0),
        (0, 2, 2),
        (0, 0, 2),
        (0, 2, 0)
    ]
    
    # Determine which (i,j,k) are covered by Strassen blocks
    covered = np.zeros((n, m, p), dtype=bool)
    for i_s, j_s, k_s in blocks:
        for i in range(i_s, i_s+2):
            for j in range(j_s, j_s+2):
                for k in range(k_s, k_s+2):
                    covered[i, j, k] = True
    
    # Count rank
    rank_strassen = 7 * len(blocks)
    rank_schoolbook = np.sum(~covered)
    total_rank = rank_strassen + rank_schoolbook
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Add Strassen blocks
    for i_s, j_s, k_s in blocks:
        for r in range(7):
            u_r = strassen_u[r]
            v_r = strassen_v[r]
            w_r = strassen_w[r]
            
            for li in range(2):
                for lj in range(2):
                    a_idx = (i_s + li) * m + (j_s + lj)
                    U_full[a_idx, col_offset] = u_r[li * 2 + lj]
                    
                    for lk in range(2):
                        b_idx = (j_s + lj) * p + (k_s + lk)
                        V_full[b_idx, col_offset] = v_r[lj * 2 + lk]
                        
                        c_idx = (k_s + lk) * n + (i_s + li)
                        W_full[c_idx, col_offset] = w_r[li * 2 + lk]
            col_offset += 1
            
    # Add Schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_full[a_idx, col_offset] = 1
                    V_full[b_idx, col_offset] = 1
                    W_full[c_idx, col_offset] = 1
                    col_offset += 1
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_full[a, :] * V_full[b, :] * W_full[c, :])
                max_err = max(max_err, abs(val - 1.0))
                
    if max_err < 1e-6:
        return {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Use Lander-Strassen rank-23 decomposition for 3x3x3
    # The 23 multiplication formulas for 3x3 matrix multiplication
    # Each formula gives one term in the decomposition
    
    # Lander-Strassen 23-multiplication algorithm for 3x3 matrices
    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    # C = A*B, C[i][k] = sum_j A[i][j]*B[j][k]
    
    # The 23 rank-1 terms (u, v, w) where C = sum_r u_r * v_r * w_r
    # u_r is a 3x3 matrix (flattened to 9), v_r is 3x3, w_r is 3x3
    # C = sum_r (A dot u_r) * (B dot v_r) * w_r ... no
    # Actually: C[i,k] = sum_r U[i,k,r] * (sum_j A[i,j]*U_a[i,j,r]) * (sum_j B[j,k]*U_b[j,k,r])
    # Wait, the interface is: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # where a=(i,j), b=(j,k), c=(i,k)
    # So we need: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for each (i,j,k)
    # This means the "product" is: for each r, we compute A[i,j]*B[j,k] weighted by U[i,j,r]*V[j,k,r]
    # and add to C[i,k] weighted by W[i,k,r]
    # So C[i,k] = sum_r W[i,k,r] * (sum_{j'} U[i,j',r] * A[i,j'] * V[j',k,r] * B[j',k] / V[j',k,r] * V[j',k,r])
    # Hmm, this isn't quite right. Let me re-read the interface.
    
    # The tensor equation is:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This means the bilinear map (A, B) -> C has the property that
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # is computed as: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Wait no. Let's be more careful.
    
    # If we define for each r:
    #   a_r[i,j] = U[i*m+j, r]
    #   b_r[j,k] = V[j*p+k, r]  
    #   c_r[i,k] = W[k*n+i, r]
    # Then the formula is:
    #   C[i,k] = sum_r c_r[i,k] * (sum_j a_r[i,j] * A[i,j]) * (sum_j b_r[j,k] * B[j,k])
    # Wait, that doesn't look right either. Let me think again.
    
    # The tensor T has entries T[(i,j),(j,k),(i,k)] = 1
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # So T[(i,j),(j,k),(i,k)] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1
    
    # The bilinear form: for input vectors A_vec and B_vec (flattened A and B),
    # the output C_vec is: C_vec[c] = sum_{a,b} T[a,b,c] * A_vec[a] * B_vec[b]
    # = sum_{a,b} sum_r U[a,r] * V[b,r] * W[c,r] * A_vec[a] * B_vec[b]
    # = sum_r W[c,r] * (sum_a U[a,r] * A_vec[a]) * (sum_b V[b,r] * B_vec[b])
    
    # So: C_vec[c] = sum_r W[c,r] * f_r(A) * g_r(B)
    # where f_r(A) = sum_a U[a,r] * A_vec[a] and g_r(B) = sum_b V[b,r] * B_vec[b]
    
    # In matrix form: f_r(A) = u_r^T * A_vec where u_r is the r-th column of U
    # g_r(B) = v_r^T * B_vec where v_r is the r-th column of V
    
    # So we need to find 23 vectors u_r (9-dim), v_r (9-dim), w_r (9-dim) such that
    # for all (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1
    
    # This is equivalent to finding a rank-23 decomposition of the 3x3x3 multiplication tensor.
    
    # Let me use a known Lander-Strassen construction.
    # I'll use a simplified version and verify it works.
    
    # Actually, let me just try a direct numerical search with a better method.
    # Use scipy's least_squares or a custom gradient-based approach.
    
    def solve_tensor_333(rank_target, max_time=50.0):
        """Solve for rank-r decomposition of 3x3x3 matrix multiplication tensor."""
        t0 = time.time()
        n3 = m3 = p3 = 3
        size = 9  # 3*3
        
        # Build target: T[a,b,c] = 1 for valid (i,j,k), 0 otherwise
        # Valid triples: a=i*3+j, b=j*3+k, c=k*3+i for i,j,k in 0..2
        # Note: not all 9x9x9 = 729 combinations are valid, only 27 are
        # The constraint is only on those 27 entries
        
        # Collect the 27 (a,b,c) indices
        abc_indices = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    abc_indices.append((a, b, c))
        
        # We need: sum_r U[a,r] * V[b,r] * W[c,r] = 1 for each (a,b,c) in abc_indices
        # This is a system of 27 equations with 3*9*rank_target = 27*rank_target unknowns
        
        # Use alternating minimization with a twist:
        # For fixed V, W: U[a,:] should satisfy sum_r U[a,r] * (V[b,r]*W[c,r]) = 1 for all b,c
        # This is: U[a,:] @ M = 1 where M[r, bc] = V[b,r]*W[c,r]
        # But we only need it for specific (b,c) pairs corresponding to valid (i,j,k)
        
        # Let me use a different approach: parameterize and use gradient descent
        # with the loss = sum over the 27 constraints of (sum_r U[a,r]*V[b,r]*W[c,r] - 1)^2
        
        best_err = float('inf')
        best_sol = None
        
        for trial in range(50):
            if time.time() - t0 > max_time:
                break
            
            np.random.seed(1000 + trial)
            U = np.random.randn(size, rank_target) * 0.3
            V = np.random.randn(size, rank_target) * 0.3
            W = np.random.randn(size, rank_target) * 0.3
            
            # Scale to make initial products closer to 1
            # For each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] should be ~1
            # With random init, typical product is ~0.3^3 * sqrt(rank) ~ 0.027*4.8 ~ 0.13
            # Scale up by ~7.7
            scale = 7.0
            U *= scale
            V *= scale
            W *= scale
            
            lr = 0.01
            for it in range(2000):
                if time.time() - t0 > max_time:
                    break
                
                # Compute gradient of loss = sum (sum_r U[a,r]*V[b,r]*W[c,r] - 1)^2
                # dL/dU[a,r] = 2 * sum over (a,b,c) with this a: (val-1) * V[b,r]*W[c,r]
                # dL/dV[b,r] = 2 * sum over (a,b,c) with this b: (val-1) * U[a,r]*W[c,r]
                # dL/dW[c,r] = 2 * sum over (a,b,c) with this c: (val-1) * U[a,r]*V[b,r]
                
                # Compute values for all 27 constraints
                vals = np.zeros(len(abc_indices))
                for idx, (a, b, c) in enumerate(abc_indices):
                    vals[idx] = np.sum(U[a, :] * V[b, :] * W[c, :])
                
                err = np.sum((vals - 1.0) ** 2)
                if err < best_err:
                    best_err = err
                    best_sol = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-10:
                    return U, V, W, np.sqrt(err)
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for idx, (a, b, c) in enumerate(abc_indices):
                    diff = 2.0 * (vals[idx] - 1.0)
                    prod = U[a, :] * V[b, :] * W[c, :]
                    dU[a, :] += diff * V[b, :] * W[c, :]
                    dV[b, :] += diff * U[a, :] * W[c, :]
                    dW[c, :] += diff * U[a, :] * V[b, :]
                
                # Gradient step with learning rate decay
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Decay learning rate
                if it % 200 == 199:
                    lr *= 0.5
                
                # Periodic restart if stuck
                if it % 500 == 499 and err > 0.1:
                    # Add small random perturbation
                    U += np.random.randn(size, rank_target) * 0.01
                    V += np.random.randn(size, rank_target) * 0.01
                    W += np.random.randn(size, rank_target) * 0.01
        
        return best_sol[0], best_sol[1], best_sol[2], np.sqrt(best_err)
    
    # Try to solve 3x3x3 at rank 23
    U3 = V3 = W3 = None
    err3 = float('inf')
    if check_deadline():
        try:
            U3, V3, W3, err3 = solve_tensor_333(23, max_time=min(60, DEADLINE - (time.time() - start_time) - 40))
            if err3 < 1e-8:
                print(f"Found 3x3x3 rank 23 solution with error {err3}")
        except Exception as e:
            print(f"Error: {e}")
            U3 = None
    
    if U3 is not None and check_deadline():
        # Build full 3x5x5 solution
        # Blocks:
        # Block 1: i in 0-2, j in 0-2, k in 0-2 -> 3x3x3, rank 23
        # Block 2: i in 0-2, j in 0-2, k in 3-4 -> 3x3x2, rank 18 (schoolbook)
        # Block 3: i in 0-2, j in 3-4, k in 0-2 -> 3x2x3, rank 18 (schoolbook)
        # Block 4: i in 0-2, j in 3-4, k in 3-4 -> 3x2x2, rank 12 (schoolbook)
        # Total: 23 + 18 + 18 + 12 = 71
        
        total_rank = 23 + 18 + 18 + 12  # 71
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        col_offset = 0
        
        # Block 1: 3x3x3
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    a_local = i * 3 + j
                    b_local = j * 3 + k
                    c_local = k * 3 + i
                    U_full[a_idx, col_offset:col_offset+23] = U3[a_local, :]
                    V_full[b_idx, col_offset:col_offset+23] = V3[b_local, :]
                    W_full[c_idx, col_offset:col_offset+23] = W3[c_local, :]
        col_offset += 23
        
        # Block 2: 3x3x2 (schoolbook)
        for i in range(3):
            for j in range(3):
                for k in range(3, 5):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_full[a_idx, col_offset] = 1
                    V_full[b_idx, col_offset] = 1
                    W_full[c_idx, col_offset] = 1
                    col_offset += 1
        
        # Block 3: 3x2x3 (schoolbook)
        for i in range(3):
            for j in range(3, 5):
                for k in range(3):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_full[a_idx, col_offset] = 1
                    V_full[b_idx, col_offset] = 1
                    W_full[c_idx, col_offset] = 1
                    col_offset += 1
        
        # Block 4: 3x2x2 (schoolbook)
        for i in range(3):
            for j in range(3, 5):
                for k in range(3, 5):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_full[a_idx, col_offset] = 1
                    V_full[b_idx, col_offset] = 1
                    W_full[c_idx, col_offset] = 1
                    col_offset += 1
        
        # Verify
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_full[a, :] * V_full[b, :] * W_full[c, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            print(f"Full solution verified with max error {max_err}, rank {total_rank}")
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
        else:
            print(f"Verification failed with error {max_err}")
    
    # Fallback: schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
