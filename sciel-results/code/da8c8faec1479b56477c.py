import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (evaluator kills at 150s)
    
    # Precompute the tensor structure
    # U index: i*m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k  (j in 0..m-1, k in 0..p-1)
    # W index: k*n + i  (k in 0..p-1, i in 0..n-1)
    
    nonzero_indices = []
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                nonzero_indices.append((u, v, w))
                nonzero_set.add((u, v, w))
    
    zero_indices = []
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                if (u, v, w) not in nonzero_set:
                    zero_indices.append((u, v, w))
    
    # Convert to arrays for vectorized operations
    nz_u = np.array([x[0] for x in nonzero_indices])
    nz_v = np.array([x[1] for x in nonzero_indices])
    nz_w = np.array([x[2] for x in nonzero_indices])
    
    z_u = np.array([x[0] for x in zero_indices])
    z_v = np.array([x[1] for x in zero_indices])
    z_w = np.array([x[2] for x in zero_indices])
    
    def calculate_error(U, V, W):
        """Calculate sum of squared errors using vectorized numpy."""
        rank = U.shape[1]
        
        # Non-zero entries: should equal 1
        # S_nz[i] = sum_r U[nz_u[i], r] * V[nz_v[i], r] * W[nz_w[i], r]
        S_nz = np.sum(U[nz_u, :] * V[nz_v, :] * W[nz_w, :], axis=1)
        err_nz = np.sum((S_nz - 1.0) ** 2)
        
        # Zero entries: should equal 0
        S_z = np.sum(U[z_u, :] * V[z_v, :] * W[z_w, :], axis=1)
        err_z = np.sum(S_z ** 2)
        
        return err_nz + err_z
    
    def calculate_gradient(U, V, W, S_nz=None, S_z=None):
        """Calculate gradient of the error function."""
        rank = U.shape[1]
        
        # S_nz[i] = sum_r U[nz_u[i], r] * V[nz_v[i], r] * W[nz_w[i], r]
        # dE/du_ur = 2 * sum_i (S_nz[i] - 1) * V[nz_v[i], r] * W[nz_w[i], r]
        #          + 2 * sum_j S_z[j] * V[z_v[j], r] * W[z_w[j], r]
        
        if S_nz is None:
            S_nz = np.sum(U[nz_u, :] * V[nz_v, :] * W[nz_w, :], axis=1)
        if S_z is None:
            S_z = np.sum(U[z_u, :] * V[z_v, :] * W[z_w, :], axis=1)
        
        err_nz = S_nz - 1.0
        err_z = S_z
        
        # For U: grad_U[u, r] = 2 * (sum_i err_nz[i] * V[nz_v[i], r] * W[nz_w[i], r] 
        #                           + sum_j err_z[j] * V[z_v[j], r] * W[z_w[j], r])
        # Only rows in nz_u or z_u have non-zero gradients
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Compute contributions for U
        # For each nonzero index i:
        #   grad_U[nz_u[i], r] += 2 * err_nz[i] * V[nz_v[i], r] * W[nz_w[i], r]
        # For each zero index j:
        #   grad_U[z_u[j], r] += 2 * err_z[j] * V[z_v[j], r] * W[z_w[j], r]
        
        # Vectorized approach:
        # Let A[i, r] = V[nz_v[i], r] * W[nz_w[i], r]
        # Then grad_U[nz_u[i], r] = 2 * err_nz[i] * A[i, r]
        
        A_nz = V[nz_v, :] * W[nz_w, :]  # shape (len(nz), rank)
        contrib_U_nz = 2.0 * err_nz[:, np.newaxis] * A_nz  # shape (len(nz), rank)
        
        # Accumulate into grad_U
        np.add.at(grad_U, (nz_u, slice(None)), contrib_U_nz)
        
        A_z = V[z_v, :] * W[z_w, :]  # shape (len(z), rank)
        contrib_U_z = 2.0 * err_z[:, np.newaxis] * A_z
        np.add.at(grad_U, (z_u, slice(None)), contrib_U_z)
        
        # For V:
        # grad_V[v, r] = 2 * (sum_i err_nz[i] * U[nz_u[i], r] * W[nz_w[i], r] 
        #                    + sum_j err_z[j] * U[z_u[j], r] * W[z_w[j], r])
        
        B_nz = U[nz_u, :] * W[nz_w, :]
        contrib_V_nz = 2.0 * err_nz[:, np.newaxis] * B_nz
        np.add.at(grad_V, (nz_v, slice(None)), contrib_V_nz)
        
        B_z = U[z_u, :] * W[z_w, :]
        contrib_V_z = 2.0 * err_z[:, np.newaxis] * B_z
        np.add.at(grad_V, (z_v, slice(None)), contrib_V_z)
        
        # For W:
        # grad_W[w, r] = 2 * (sum_i err_nz[i] * U[nz_u[i], r] * V[nz_v[i], r] 
        #                    + sum_j err_z[j] * U[z_u[j], r] * V[z_v[j], r])
        
        C_nz = U[nz_u, :] * V[nz_v, :]
        contrib_W_nz = 2.0 * err_nz[:, np.newaxis] * C_nz
        np.add.at(grad_W, (nz_w, slice(None)), contrib_W_nz)
        
        C_z = U[z_u, :] * V[z_v, :]
        contrib_W_z = 2.0 * err_z[:, np.newaxis] * C_z
        np.add.at(grad_W, (z_w, slice(None)), contrib_W_z)
        
        return grad_U, grad_V, grad_W, S_nz, S_z
    
    def discrete_local_search(rank, max_restarts=10, max_iters=5000):
        if time.time() > deadline:
            return None
            
        # Discrete values to search over
        vals = [0.0, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5]
        
        best_overall = None
        best_overall_err = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize: start with random assignment, biased towards small values
            # Use a smarter initialization: random but with some structure
            U = np.random.choice(vals, size=(n*m, rank), p=[0.3, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05])
            V = np.random.choice(vals, size=(m*p, rank), p=[0.3, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05])
            W = np.random.choice(vals, size=(p*n, rank), p=[0.3, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05])
            
            curr_err = calculate_error(U, V, W)
            best_err = curr_err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_err < best_overall_err:
                best_overall_err = best_err
                best_overall = (best_U.copy(), best_V.copy(), best_W.copy())
                if best_err < 1e-9:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            
            stuck = 0
            no_improve_count = 0
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute gradient to guide the search
                grad_U, grad_V, grad_W, S_nz, S_z = calculate_gradient(U, V, W)
                
                # Choose which matrix to modify based on gradient magnitude
                grad_U_norm = np.sum(grad_U ** 2)
                grad_V_norm = np.sum(grad_V ** 2)
                grad_W_norm = np.sum(grad_W ** 2)
                
                total = grad_U_norm + grad_V_norm + grad_W_norm
                if total < 1e-15:
                    # At a local minimum
                    if stuck < 50:
                        stuck += 1
                        # Try a random perturbation to escape
                        mat_type = random.choice(['U', 'V', 'W'])
                        if mat_type == 'U':
                            row = random.randint(0, n*m - 1)
                            col = random.randint(0, rank - 1)
                            old_val = U[row, col]
                            new_val = random.choice([v for v in vals if v != old_val])
                            U[row, col] = new_val
                            new_err = calculate_error(U, V, W)
                            if new_err < curr_err * 1.01 or random.random() < 0.1:  # Accept slightly worse
                                curr_err = new_err
                                stuck = 0
                            else:
                                U[row, col] = old_val
                        elif mat_type == 'V':
                            row = random.randint(0, m*p - 1)
                            col = random.randint(0, rank - 1)
                            old_val = V[row, col]
                            new_val = random.choice([v for v in vals if v != old_val])
                            V[row, col] = new_val
                            new_err = calculate_error(U, V, W)
                            if new_err < curr_err * 1.01 or random.random() < 0.1:
                                curr_err = new_err
                                stuck = 0
                            else:
                                V[row, col] = old_val
                        else:
                            row = random.randint(0, p*n - 1)
                            col = random.randint(0, rank - 1)
                            old_val = W[row, col]
                            new_val = random.choice([v for v in vals if v != old_val])
                            W[row, col] = new_val
                            new_err = calculate_error(U, V, W)
                            if new_err < curr_err * 1.01 or random.random() < 0.1:
                                curr_err = new_err
                                stuck = 0
                            else:
                                W[row, col] = old_val
                        continue
                    else:
                        break
                
                # Select matrix proportional to gradient norm
                r = random.random() * total
                if r < grad_U_norm:
                    mat_type = 'U'
                    mat = U
                    rows = n*m
                    grad = grad_U
                elif r < grad_U_norm + grad_V_norm:
                    mat_type = 'V'
                    mat = V
                    rows = m*p
                    grad = grad_V
                else:
                    mat_type = 'W'
                    mat = W
                    rows = p*n
                    grad = grad_W
                
                # Select row and column, biased towards large gradient
                # Flatten gradient and pick with probability proportional to |grad|
                grad_flat = np.abs(grad).flatten()
                grad_flat = grad_flat + 1e-10  # Avoid zero
                probs = grad_flat / np.sum(grad_flat)
                
                idx = np.random.choice(len(probs), p=probs)
                row = idx // rank
                col = idx % rank
                
                old_val = mat[row, col]
                
                # Determine direction from gradient
                g = grad[row, col]
                if g > 0:
                    # Increase value to reduce error
                    candidates = [v for v in vals if v > old_val]
                elif g < 0:
                    # Decrease value to reduce error
                    candidates = [v for v in vals if v < old_val]
                else:
                    candidates = [v for v in vals if v != old_val]
                
                if not candidates:
                    candidates = [v for v in vals if v != old_val]
                
                # Try the candidate closest to old_val in the correct direction
                # Actually, try all candidates and pick the best
                best_candidate = old_val
                best_new_err = curr_err
                
                for new_val in candidates:
                    mat[row, col] = new_val
                    new_err = calculate_error(U, V, W)
                    if new_err < best_new_err:
                        best_new_err = new_err
                        best_candidate = new_val
                    mat[row, col] = old_val
                
                mat[row, col] = best_candidate
                new_err = calculate_error(U, V, W)
                
                if new_err < curr_err:
                    curr_err = new_err
                    stuck = 0
                    no_improve_count = 0
                    if new_err < best_err:
                        best_err = new_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        
                        if best_err < 1e-9:
                            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                else:
                    # Revert
                    mat[row, col] = old_val
                    stuck += 1
                    no_improve_count += 1
                    
                if stuck > 200:
                    break
                if no_improve_count > 500:
                    break
            
            # After each restart, update best overall
            if best_err < best_overall_err:
                best_overall_err = best_err
                best_overall = (best_U.copy(), best_V.copy(), best_W.copy())
                if best_err < 1e-9:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        return None
    
    def polish(U, V, W):
        """Final high-precision polish pass to find exact integer/half-integer solution."""
        if time.time() > deadline:
            return None
        
        vals = [0.0, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5, 2.0, -2.0, 2.5, -2.5]
        rank = U.shape[1]
        
        curr_err = calculate_error(U, V, W)
        best_err = curr_err
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        if best_err < 1e-9:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        for it in range(20000):
            if time.time() > deadline:
                break
            
            # Try single element changes
            mat_type = random.choice(['U', 'V', 'W'])
            if mat_type == 'U':
                mat = U
                rows = n*m
            elif mat_type == 'V':
                mat = V
                rows = m*p
            else:
                mat = W
                rows = p*n
                
            row = random.randint(0, rows - 1)
            col = random.randint(0, rank - 1)
            old_val = mat[row, col]
            
            if old_val not in vals:
                # Snap to nearest valid value
                old_val = round(old_val * 2) / 2
            
            new_val = random.choice([v for v in vals if v != old_val])
            
            mat[row, col] = new_val
            new_err = calculate_error(U, V, W)
            
            if new_err < curr_err:
                curr_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if best_err < 1e-9:
                        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            else:
                mat[row, col] = old_val
            
            # Also try snapping all values to nearest half-integer
            if it % 1000 == 0:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                curr_err = calculate_error(U, V, W)
                if curr_err < best_err:
                    best_err = curr_err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if best_err < 1e-9:
                        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        if best_err < 1e-9:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        return None
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1.0
                    V[v, col] = 1.0
                    W[w, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better solution using structured initializations
    def structured_init(rank):
        """Create a structured initial guess using Fourier-like patterns."""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Try to use a cyclic structure
        for r in range(rank):
            phase = 2 * np.pi * r / rank
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    U[u, r] = 1.0
            
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    V[v, r] = 1.0
            
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    W[w, r] = 1.0
        
        # Add some randomness
        for r in range(rank):
            if random.random() < 0.3:
                U[:, r] = np.random.choice([0.0, 0.5, 1.0, -0.5, -1.0], size=n*m)
            if random.random() < 0.3:
                V[:, r] = np.random.choice([0.0, 0.5, 1.0, -0.5, -1.0], size=m*p)
            if random.random() < 0.3:
                W[:, r] = np.random.choice([0.0, 0.5, 1.0, -0.5, -1.0], size=p*n)
        
        return U, V, W
    
    # Search from low rank to high rank
    # Known lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Let's try ranks from 15 to 36
    
    best_result = None
    best_rank = n * m * p
    
    for rank in range(15, n * m * p + 1):
        if time.time() > deadline:
            break
        
        # Try multiple restarts with different initializations
        found = False
        for init_method in range(3):
            if time.time() > deadline:
                break
            
            if init_method == 0:
                # Random initialization
                result = discrete_local_search(rank, max_restarts=5, max_iters=3000)
            elif init_method == 1:
                # Structured initialization
                U, V, W = structured_init(rank)
                curr_err = calculate_error(U, V, W)
                if curr_err < 1e-9:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                # Polish from structured init
                result = polish(U, V, W)
                if result is None:
                    result = discrete_local_search(rank, max_restarts=3, max_iters=2000)
            else:
                # Hybrid: start from schoolbook and remove columns
                if rank < n * m * p:
                    U_s = np.zeros((n*m, n*m*p))
                    V_s = np.zeros((m*p, n*m*p))
                    W_s = np.zeros((p*n, n*m*p))
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u = i * m + j
                                v = j * p + k
                                w = k * n + i
                                U_s[u, col] = 1.0
                                V_s[v, col] = 1.0
                                W_s[w, col] = 1.0
                                col += 1
                    # Randomly select 'rank' columns
                    cols = np.random.choice(n*m*p, size=rank, replace=False)
                    U = U_s[:, cols]
                    V = V_s[:, cols]
                    W = W_s[:, cols]
                    curr_err = calculate_error(U, V, W)
                    if curr_err < 1e-9:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    result = polish(U, V, W)
                    if result is None:
                        result = discrete_local_search(rank, max_restarts=3, max_iters=2000)
                else:
                    result = None
            
            if result is not None:
                found = True
                break
        
        if found:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
