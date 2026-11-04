"""Matrix multiplication 2x3 x 3x4 - rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    U_dims = n * m  # 6
    V_dims = m * p  # 12
    W_dims = p * n  # 8

    # Build the matrix multiplication tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_dense = np.zeros((U_dims, V_dims, W_dims))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_dense[a, b, c] = 1.0

    def compute_error_full(U, V, W):
        """Compute the squared Frobenius norm of the reconstruction error and return T_rec."""
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        diff = T_rec - T_dense
        return np.sum(diff * diff), T_rec

    def snap_and_verify(U, V, W, tol=1e-6):
        """Snap to half-integers and verify exactness."""
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        # Check if snapping was close
        if (np.linalg.norm(U_snap - U) > 1e-3 or 
            np.linalg.norm(V_snap - V) > 1e-3 or 
            np.linalg.norm(W_snap - W) > 1e-3):
            return None

        # Verify exact decomposition with snapped values
        T_rec = np.einsum('at,bt,ct->abc', U_snap, V_snap, W_snap)
        err = np.linalg.norm(T_rec - T_dense)
        if err < 1e-8:
            return U_snap, V_snap, W_snap
        return None

    def to_list(mat):
        """Convert numpy array to list of lists."""
        return [[float(x) for x in row] for row in mat]

    def discrete_local_search(rank, n_restarts=10, max_steps=10000, seed=0):
        """
        Fast discrete local search with incremental error updates.
        """
        rng = random.Random(seed)
        best_err = float('inf')
        best_UVW = None
        
        # Values to try: half-integers in [-1.5, 1.5]
        values = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Random initialization
            U = np.array([[rng.choice(values) for _ in range(rank)] for _ in range(U_dims)])
            V = np.array([[rng.choice(values) for _ in range(rank)] for _ in range(V_dims)])
            W = np.array([[rng.choice(values) for _ in range(rank)] for _ in range(W_dims)])
            
            current_err, T_rec = compute_error_full(U, V, W)
            
            # Hill Climbing with incremental updates
            improved = True
            steps = 0
            while improved and steps < max_steps and time.time() < deadline:
                improved = False
                steps += 1
                
                # Randomly select a coordinate to update to avoid getting stuck
                mat_choice = rng.randint(0, 2)
                
                if mat_choice == 0:
                    t = rng.randint(0, rank - 1)
                    a = rng.randint(0, U_dims - 1)
                    old_val = U[a, t]
                    
                    # Precompute outer product for this t
                    outer_VW = np.outer(V[:, t], W[:, t])
                    
                    best_new_val = old_val
                    best_new_err = current_err
                    
                    for new_val in values:
                        if new_val == old_val:
                            continue
                        delta = new_val - old_val
                        
                        # Current slice error
                        diff_old = T_rec[a, :, :] - T_dense[a, :, :]
                        err_old_slice = np.sum(diff_old * diff_old)
                        
                        # New slice
                        T_rec_new_slice = T_rec[a, :, :] + delta * outer_VW
                        diff_new = T_rec_new_slice - T_dense[a, :, :]
                        err_new_slice = np.sum(diff_new * diff_new)
                        
                        new_err = current_err - err_old_slice + err_new_slice
                        
                        if new_err < best_new_err:
                            best_new_err = new_err
                            best_new_val = new_val

                    if best_new_val != old_val:
                        delta = best_new_val - old_val
                        U[a, t] = best_new_val
                        T_rec[a, :, :] += delta * outer_VW
                        current_err = best_new_err
                        improved = True

                elif mat_choice == 1:
                    t = rng.randint(0, rank - 1)
                    b = rng.randint(0, V_dims - 1)
                    old_val = V[b, t]
                    
                    # Precompute outer product for this t
                    outer_UW = np.outer(U[:, t], W[:, t])
                    
                    best_new_val = old_val
                    best_new_err = current_err
                    
                    for new_val in values:
                        if new_val == old_val:
                            continue
                        delta = new_val - old_val
                        
                        # Current slice error
                        diff_old = T_rec[:, b, :] - T_dense[:, b, :]
                        err_old_slice = np.sum(diff_old * diff_old)
                        
                        # New slice
                        T_rec_new_slice = T_rec[:, b, :] + delta * outer_UW
                        diff_new = T_rec_new_slice - T_dense[:, b, :]
                        err_new_slice = np.sum(diff_new * diff_new)
                        
                        new_err = current_err - err_old_slice + err_new_slice
                        
                        if new_err < best_new_err:
                            best_new_err = new_err
                            best_new_val = new_val

                    if best_new_val != old_val:
                        delta = best_new_val - old_val
                        V[b, t] = best_new_val
                        T_rec[:, b, :] += delta * outer_UW
                        current_err = best_new_err
                        improved = True

                else:
                    t = rng.randint(0, rank - 1)
                    c = rng.randint(0, W_dims - 1)
                    old_val = W[c, t]
                    
                    # Precompute outer product for this t
                    outer_UV = np.outer(U[:, t], V[:, t])
                    
                    best_new_val = old_val
                    best_new_err = current_err
                    
                    for new_val in values:
                        if new_val == old_val:
                            continue
                        delta = new_val - old_val
                        
                        # Current slice error
                        diff_old = T_rec[:, :, c] - T_dense[:, :, c]
                        err_old_slice = np.sum(diff_old * diff_old)
                        
                        # New slice
                        T_rec_new_slice = T_rec[:, :, c] + delta * outer_UV
                        diff_new = T_rec_new_slice - T_dense[:, :, c]
                        err_new_slice = np.sum(diff_new * diff_new)
                        
                        new_err = current_err - err_old_slice + err_new_slice
                        
                        if new_err < best_new_err:
                            best_new_err = new_err
                            best_new_val = new_val

                    if best_new_val != old_val:
                        delta = best_new_val - old_val
                        W[c, t] = best_new_val
                        T_rec[:, :, c] += delta * outer_UV
                        current_err = best_new_err
                        improved = True

            if current_err < best_err:
                best_err = current_err
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-8:
                break

        if best_UVW is not None:
            U, V, W = best_UVW
            result = snap_and_verify(U, V, W)
            if result:
                return {'U': to_list(result[0]), 'V': to_list(result[1]), 'W': to_list(result[2])}
        return None

    # Try ranks from the theoretical lower bound (19) up to schoolbook (24)
    for r in range(19, 25):
        if time.time() > deadline - 10:
            break
        
        found = None
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            found = discrete_local_search(r, n_restarts=10, max_steps=5000, seed=r*100+seed)
            if found:
                break
        
        if found:
            return found

    # Fallback: Schoolbook decomposition (Rank 24)
    rank = n * m * p
    U = [[0] * rank for _ in range(U_dims)]
    V = [[0] * rank for _ in range(V_dims)]
    W = [[0] * rank for _ in range(W_dims)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
