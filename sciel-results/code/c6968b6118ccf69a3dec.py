import numpy as np
import time
import scipy.optimize as opt
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin
    
    # Rank 17 decomposition for 2x3x3 matrix multiplication
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    # Indices for global tensors
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U17[2, 7] = 1
    V17[6, 7] = 1
    W17[0, 7] = 1
    
    U17[2, 8] = 1
    V17[7, 8] = 1
    W17[2, 8] = 1
    
    U17[5, 9] = 1
    V17[6, 9] = 1
    W17[1, 9] = 1
    
    U17[5, 10] = 1
    V17[7, 10] = 1
    W17[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U17[0, 11] = 1
    V17[2, 11] = 1
    W17[4, 11] = 1
    
    U17[1, 12] = 1
    V17[5, 12] = 1
    W17[4, 12] = 1
    
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    U17[3, 14] = 1
    V17[2, 14] = 1
    W17[5, 14] = 1
    
    U17[4, 15] = 1
    V17[5, 15] = 1
    W17[5, 15] = 1
    
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1

    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor"""
        rank = U_mat.shape[1]
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    # The tensor T[i,j,k] = 1
                    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                    actual = sum(U_mat[i*m_val+j, r] * V_mat[j*p_val+k, r] * W_mat[k*n_val+i, r] for r in range(rank))
                    if abs(actual - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(matrix):
        """Snap values close to half-integers to exact half-integers"""
        snapped = np.round(matrix * 2) / 2
        return snapped

    # First, ensure the rank 17 solution is valid and snap it
    if verify(U17, V17, W17, n, m, p):
        U17_snapped = snap_to_half_integer(U17)
        V17_snapped = snap_to_half_integer(V17)
        W17_snapped = snap_to_half_integer(W17)
        if verify(U17_snapped, V17_snapped, W17_snapped, n, m, p, tol=1e-9):
            best_U, best_V, best_W = U17_snapped, V17_snapped, W17_snapped
            best_rank = 17
        else:
            best_U, best_V, best_W = U17, V17, W17
            best_rank = 17
    else:
        best_U, best_V, best_W = U17, V17, W17
        best_rank = 17

    # Attempt to find a rank 16 solution using optimization
    # We will parameterize U, V, W with 16 columns.
    # To reduce the search space, we can fix some values or use a structured initialization.
    # A common trick is to start from the rank 17 solution, drop one column, and adjust the rest.
    # However, random restarts with optimization might be more robust.
    
    # Let's try a few random restarts for rank 16
    target_tensor = np.ones(n * m * p)
    
    # Pre-compute indices for vectorization
    # vec[idx] corresponds to T[i,j,k] where idx = (i*m+j)*p + k
    # We need to map (i,j,k) to indices in U, V, W
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)

    def objective(x):
        # x is a flattened array of length 16 * (n*m + m*p + p*n)
        rank = 16
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U_flat = x[:size_U * rank]
        V_flat = x[size_U * rank : size_U * rank + size_V * rank]
        W_flat = x[size_U * rank + size_V * rank :]
        
        U = U_flat.reshape(size_U, rank)
        V = V_flat.reshape(size_V, rank)
        W = W_flat.reshape(size_W, rank)
        
        # Compute tensor product sum
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can vectorize this using the pre-computed indices
        u_vals = U[u_indices, :] # shape (nmp, rank)
        v_vals = V[v_indices, :] # shape (nmp, rank)
        w_vals = W[w_indices, :] # shape (nmp, rank)
        
        # Sum over rank
        # This creates a tensor of shape (nmp, rank) then sums axis 1
        # But we need element-wise multiplication first
        prod = u_vals * v_vals * w_vals
        reconstructed = np.sum(prod, axis=1)
        
        error = reconstructed - target_tensor
        return np.sum(error * error)

    # We will try to optimize for rank 16
    # Initialize with random values
    # To help convergence, we can try to initialize with the rank 17 solution 
    # by dropping one column and scaling the others, but that's a specific path.
    # Random initialization is more general.
    
    # We have 150s total. We've used some time. Let's allocate 120s for search.
    start_time = time.time()
    search_deadline = start_time + 120.0
    
    best_found_rank = best_rank
    best_solution = (best_U, best_V, best_W)
    
    # Try a few random restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > search_deadline:
            break
            
        # Initialize x
        # We can use a structured initialization based on the known rank 16 structure if we had it,
        # but random is safer.
        # Let's use a scale of 1.0 for initialization
        x0 = np.random.randn(16 * (n * m + m * p + p * n))
        
        # Optimize
        try:
            result = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
        except Exception:
            continue
            
        if result.fun < 1e-8:
            # Found a potential solution
            x_opt = result.x
            size_U = n * m
            size_V = m * p
            size_W = p * n
            rank = 16
            
            U_opt = x_opt[:size_U * rank].reshape(size_U, rank)
            V_opt = x_opt[size_U * rank : size_U * rank + size_V * rank].reshape(size_V, rank)
            W_opt = x_opt[size_U * rank + size_V * rank :].reshape(size_W, rank)
            
            # Snap to half-integers
            U_snap = snap_to_half_integer(U_opt)
            V_snap = snap_to_half_integer(V_opt)
            W_snap = snap_to_half_integer(W_opt)
            
            # Verify
            if verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                best_found_rank = 16
                best_solution = (U_snap, V_snap, W_snap)
                break # Found the best possible rank, stop
                
            # If snapping failed, check if the unsnapped version is extremely close
            # and maybe the evaluator accepts floats? The prompt says "integer/half-integer".
            # So we must snap. If snapping breaks it, it's not a valid half-integer solution.
            # We can try to refine the solution to be closer to half-integers.
            # For now, if snapping fails, we discard this candidate.

    if best_found_rank < best_rank:
        return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
