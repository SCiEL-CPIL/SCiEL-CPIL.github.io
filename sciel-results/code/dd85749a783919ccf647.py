import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin before 140s limit

    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    # Build the target tensor for 2x3x6 multiplication
    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))

    # Schoolbook solution: rank = n*m*p = 36
    best_rank = n * m * p
    rank = best_rank
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Strategy: Split the 3x6 matrix into two 3x3 blocks
    # B = [B1 | B2] where B1 = B[:, 0:3], B2 = B[:, 3:6]
    # C = A*B1 + A*B2 = C1 + C2
    
    # Sub-problem 1: k in {0,1,2}
    # a = i*3+j, b = j*3+k, c_sub = 2*k+i
    # Sub-problem 2: k in {3,4,5}
    # a = i*3+j, b = j*3+(k-3), c_sub = 2*(k-3)+i
    
    sub_valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(3):
                a = i * m + j
                b = j * 3 + k
                c_sub = 2 * k + i
                sub_valid_triples.add((a, b, c_sub))
    
    sub_num_A = 6
    sub_num_B = 9
    sub_num_C = 6

    # Try to find low-rank decomposition for 2x3x3 sub-problem using ALS
    # Known rank is 16. Schoolbook is 18.
    
    found_sub = False
    best_sub_rank = 18
    best_sub_U = best_sub_V = best_sub_W = None

    for target_sub_rank in [16, 17]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for restart in range(10):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(1000 * target_sub_rank + restart)
            U_curr = np.random.randn(sub_num_A, target_sub_rank)
            V_curr = np.random.randn(sub_num_B, target_sub_rank)
            W_curr = np.random.randn(sub_num_C, target_sub_rank)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U: U[a, :] * (V[b, :] * W[c, :]) = 1 for valid (a,b,c)
                # For each a, we have a linear system U[a, :] X_a = y_a
                # where X_a is a matrix of rows V[b, :] * W[c, :] for valid (a,b,c)
                # and y_a is a vector of 1s.
                
                # Efficiently, we can stack all equations.
                # Let's build the design matrix for all valid triples.
                # This is small enough to do directly.
                
                # Actually, standard ALS fixes two matrices and solves for the third.
                # Fix V, W. Solve for U.
                # For each a in range(sub_num_A):
                #   Rows: for each (a,b,c) in valid_triples, row = V[b] * W[c]
                #   Target: 1
                #   Solve min || U[a] * X_a^T - y_a ||^2
                #   U[a] = (X_a X_a^T)^{-1} X_a y_a^T
                #   Note: U[a] is a row vector of length r.
                #   Let X_a be matrix of size (num_valid_for_a) x r.
                #   Then U[a] = y_a^T X_a^T (X_a X_a^T)^{-1}
                
                # We can vectorize this by building a global system or solving per-a.
                # Per-a is safer.
                
                U_new = np.zeros_like(U_curr)
                for a in range(sub_num_A):
                    X_a = []
                    y_a = []
                    for (aa, b, c) in sub_valid_triples:
                        if aa == a:
                            # Element-wise product of V[b] and W[c]
                            X_a.append(V_curr[b] * W_curr[c])
                            y_a.append(1.0)
                    if X_a:
                        X_a = np.array(X_a)
                        y_a = np.array(y_a)
                        # Solve min || X_a^T U[a]^T - y_a ||^2
                        # U[a] is 1 x r. Let u = U[a]^T (r x 1).
                        # min || X_a u - y_a ||^2
                        # u = (X_a^T X_a)^{-1} X_a^T y_a
                        # Use lstsq for stability
                        u, residuals, rank_u, s = np.linalg.lstsq(X_a, y_a, rcond=None)
                        U_new[a] = u
                    else:
                        U_new[a] = 0.2
                U_curr = U_new

                # Solve for V
                V_new = np.zeros_like(V_curr)
                for b in range(sub_num_B):
                    X_b = []
                    y_b = []
                    for (a, bb, c) in sub_valid_triples:
                        if bb == b:
                            X_b.append(U_curr[a] * W_curr[c])
                            y_b.append(1.0)
                    if X_b:
                        X_b = np.array(X_b)
                        y_b = np.array(y_b)
                        v, residuals, rank_v, s = np.linalg.lstsq(X_b, y_b, rcond=None)
                        V_new[b] = v
                    else:
                        V_new[b] = 0.2
                V_curr = V_new

                # Solve for W
                W_new = np.zeros_like(W_curr)
                for c in range(sub_num_C):
                    X_c = []
                    y_c = []
                    for (a, b, cc) in sub_valid_triples:
                        if cc == c:
                            X_c.append(U_curr[a] * V_curr[b])
                            y_c.append(1.0)
                    if X_c:
                        X_c = np.array(X_c)
                        y_c = np.array(y_c)
                        w, residuals, rank_w, s = np.linalg.lstsq(X_c, y_c, rcond=None)
                        W_new[c] = w
                    else:
                        W_new[c] = 0.2
                W_curr = W_new

                # Check convergence / validity
                # Calculate error on valid triples
                max_err = 0
                for (a, b, c) in sub_valid_triples:
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                
                # Check invalid triples (sample)
                np.random.seed(42)
                for _ in range(50):
                    a = np.random.randint(0, sub_num_A)
                    b = np.random.randint(0, sub_num_B)
                    c = np.random.randint(0, sub_num_C)
                    if (a, b, c) not in sub_valid_triples:
                        val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                        max_err = max(max_err, abs(val))
                        
                if max_err < 1e-8:
                    found = True
                    break
            
            if found:
                best_sub_rank = target_sub_rank
                best_sub_U, best_sub_V, best_sub_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                found_sub = True
                break
        
        if found_sub:
            break
    
    # If we found a sub-decomposition, construct the full 2x3x6 decomposition
    if found_sub:
        sub_rank = best_sub_rank
        full_rank = 2 * sub_rank
        
        U_full = np.zeros((num_A, full_rank))
        V_full = np.zeros((num_B, full_rank))
        W_full = np.zeros((num_C, full_rank))
        
        # First block: k in {0,1,2}
        for r in range(sub_rank):
            U_full[:, r] = best_sub_U[:, r]
            for j in range(3):
                for k in range(3):
                    sub_b = j * 3 + k
                    full_b = j * 6 + k
                    V_full[full_b, r] = best_sub_V[sub_b, r]
            for k in range(3):
                for i in range(2):
                    sub_c = 2 * k + i
                    full_c = 2 * k + i
                    W_full[full_c, r] = best_sub_W[sub_c, r]
        
        # Second block: k in {3,4,5}
        offset = sub_rank
        for r in range(sub_rank):
            U_full[:, offset + r] = best_sub_U[:, r]
            for j in range(3):
                for k_sub in range(3):
                    k = k_sub + 3
                    sub_b = j * 3 + k_sub
                    full_b = j * 6 + k
                    V_full[full_b, offset + r] = best_sub_V[sub_b, r]
            for k_sub in range(3):
                k = k_sub + 3
                for i in range(2):
                    sub_c = 2 * k_sub + i
                    full_c = 2 * k + i
                    W_full[full_c, offset + r] = best_sub_W[sub_c, r]
        
        # Verify the full decomposition
        max_err = 0
        for (a, b, c) in valid_triples:
            val = np.sum(U_full[a, :] * V_full[b, :] * W_full[c, :])
            max_err = max(max_err, abs(val - 1.0))
        
        # Check invalid positions
        for a in range(num_A):
            for b in range(num_B):
                for c in range(num_C):
                    if (a, b, c) not in valid_triples:
                        val = np.sum(U_full[a, :] * V_full[b, :] * W_full[c, :])
                        max_err = max(max_err, abs(val))
        
        if max_err < 1e-8:
            best_rank = full_rank
            best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
