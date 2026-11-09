import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_min = max(n * m, n * p, m * p) # 30
    
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    fallback = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Try to find rank 30 solution
    R = rank_min
    U = np.random.randn(n * m, R)
    V = np.random.randn(m * p, R)
    W = np.random.randn(p * n, R)
    
    # Scale down to avoid overflow
    U *= 0.1
    V *= 0.1
    W *= 0.1

    def verify(U, V, W, R):
        # Check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i', r] == delta(i, i') delta(k, k')
        # Note: W index is k'*n + i'
        # The equation is for all i, j, k, i', k'
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = delta(i, i') * delta(k, k')
        # We can check this by reconstructing the tensor and comparing.
        # Or just spot check.
        # For speed, check all.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute vector in output space
                    # vec[l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[l, r]
                    # vec should be e_{i, k} where index is k*n+i? 
                    # Wait, W index is k'*n + i'.
                    # The output C[i', k'] gets contribution from A[i, j] B[j, k] only if i'=i and k'=k.
                    # So the vector in output space should have 1 at index k*n+i and 0 elsewhere.
                    vec = np.zeros(p * n)
                    for r in range(R):
                        u_val = U[i * m + j, r]
                        v_val = V[j * p + k, r]
                        if u_val != 0 and v_val != 0:
                            vec += u_val * v_val * W[:, r]
                    # Check vec
                    target_idx = k * n + i
                    if abs(vec[target_idx] - 1.0) > 1e-5:
                        return False
                    for l in range(p * n):
                        if l != target_idx and abs(vec[l]) > 1e-5:
                            return False
        return True

    # ALS loop
    for it in range(50):
        if time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                # Solve for u = U[i*m+j, :]
                # A rows: (k, i', k'), cols: r
                # A[(k, i', k'), r] = V[j*p+k, r] * W[k'*n+i', r]
                # b[(k, i', k')] = delta(i, i') * delta(k, k')
                A_mat = np.zeros((p * p * n, R)) # p*k, n*i', p*k'
                b_vec = np.zeros(p * p * n)
                idx = 0
                for k in range(p):
                    for kp in range(p): # k'
                        for ip in range(n): # i'
                            # A entry
                            row_val = V[j * p + k, :] * W[kp * n + ip, :]
                            A_mat[idx, :] = row_val
                            # b entry
                            if k == kp and i == ip:
                                b_vec[idx] = 1.0
                            idx += 1
                # Solve least squares
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                # Solve for v = V[j*p+k, :]
                # A rows: (i, j', i', k')? No.
                # Equation: sum_r U[i*m+j', r] * V[j*p+k, r] * W[k'*n+i', r] = delta(i, i') delta(k, k')
                # Wait, the input indices are i, j for A and j, k for B.
                # In the update for V[j, k], the fixed indices are j, k.
                # The varying indices are i (from A), and the output indices i', k'.
                # But A is indexed by i, j. So i varies.
                # B is indexed by j, k. j, k fixed.
                # So the term is A[i, j] B[j, k].
                # Contribution to C[i', k'] is delta(i, i') delta(k, k').
                # So for fixed j, k, we sum over r: U[i*m+j, r] * v_r * W[k'*n+i', r] = delta(i, i') delta(k, k').
                # This must hold for all i, i', k'.
                # A rows: (i, i', k'), cols: r.
                # A[(i, i', k'), r] = U[i*m+j, r] * W[k'*n+i', r]
                # b[(i, i', k')] = delta(i, i') * delta(k, k')
                A_mat = np.zeros((n * n * p, R))
                b_vec = np.zeros(n * n * p)
                idx = 0
                for i in range(n):
                    for ip in range(n):
                        for kp in range(p):
                            row_val = U[i * m + j, :] * W[kp * n + ip, :]
                            A_mat[idx, :] = row_val
                            if i == ip and k == kp:
                                b_vec[idx] = 1.0
                            idx += 1
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass

        # Update W
        for kp in range(p): # k'
            for ip in range(n): # i'
                # Solve for w = W[kp*n+ip, :]
                # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * w_r = delta(i, ip) delta(k, kp)
                # Fixed: ip, kp.
                # Varying: i, j, k.
                # A rows: (i, j, k), cols: r.
                # A[(i, j, k), r] = U[i*m+j, r] * V[j*p+k, r]
                # b[(i, j, k)] = delta(i, ip) * delta(k, kp)
                A_mat = np.zeros((n * m * p, R))
                b_vec = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row_val = U[i * m + j, :] * V[j * p + k, :]
                            A_mat[idx, :] = row_val
                            if i == ip and k == kp:
                                b_vec[idx] = 1.0
                            idx += 1
                try:
                    W[kp * n + ip, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Check verification periodically
        if it % 10 == 0 and verify(U, V, W, R):
            # Round to half integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if verify(U_rounded, V_rounded, W_rounded, R):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}

    return fallback
# EVOLVE-BLOCK-END
