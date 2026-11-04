import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(6):
        s = b // 3
        k_block = b % 3
        
        for t in range(7):
            c = 7 * b + t
            
            u_coeff = strassen_A[t]
            for i in range(2):
                for j_loc in range(2):
                    idx_A = i * 2 + j_loc
                    coeff = u_coeff[idx_A]
                    if coeff != 0:
                        global_i = i
                        global_j = 2 * s + j_loc
                        U[global_i * m + global_j][c] = coeff
                        
            v_coeff = strassen_B[t]
            for j_loc in range(2):
                for k_loc in range(2):
                    idx_B = j_loc * 2 + k_loc
                    coeff = v_coeff[idx_B]
                    if coeff != 0:
                        global_j = 2 * s + j_loc
                        global_k = 2 * k_block + k_loc
                        V[global_j * p + global_k][c] = coeff
                        
            w_coeff = w_coeffs[t]
            for i in range(2):
                for k_loc in range(2):
                    idx_W = i * 2 + k_loc
                    coeff = w_coeff[idx_W]
                    if coeff != 0:
                        global_k = 2 * k_block + k_loc
                        global_i = i
                        W[global_k * n + global_i][c] = coeff
                        
    # Now check for linear dependencies among the 42 rank-1 tensors
    # Flatten each rank-1 tensor into a vector and check if they're linearly independent
    U_arr = np.array(U, dtype=float)  # (8, 42)
    V_arr = np.array(V, dtype=float)  # (24, 42)
    W_arr = np.array(W, dtype=float)  # (12, 42)
    
    # Each rank-1 tensor t is U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # Flatten to a vector of length 8*24*12 = 2304
    # Stack as rows of a 42 x 2304 matrix
    flat_tensors = []
    for t in range(42):
        vec = np.outer(np.outer(U_arr[:, t], V_arr[:, t]), W_arr[:, t]).flatten()
        flat_tensors.append(vec)
    
    tensor_matrix = np.array(flat_tensors)  # (42, 2304)
    
    # Compute rank of this matrix
    tol = 1e-10
    try:
        svd_vals = np.linalg.svd(tensor_matrix, compute_uv=False)
        tensor_rank = np.sum(svd_vals > tol)
    except:
        tensor_rank = 42
    
    if tensor_rank < 42:
        # Find a basis for the row space and reduce
        # Use QR decomposition with column pivoting to find independent columns
        # Actually, we need to find which rows are independent
        # Use SVD: keep the top tensor_rank singular vectors
        U_svd, S_vals, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
        # The first tensor_rank rows of U_svd give a basis
        # But we need to express the original tensors as combinations of the basis
        # Simpler: find a subset of independent columns using Gaussian elimination
        # Use row-reduced echelon form approach
        # Keep the first tensor_rank linearly independent vectors
        kept = []
        basis = []
        for t in range(42):
            vec = tensor_matrix[t]
            # Check if vec is in span of basis
            if not basis:
                kept.append(t)
                basis.append(vec.copy())
            else:
                # Try to reduce vec by basis
                remaining = vec.copy()
                independent = True
                for b_vec in basis:
                    # Project out component along b_vec
                    dot = np.dot(remaining, b_vec)
                    norm_sq = np.dot(b_vec, b_vec)
                    if norm_sq > tol:
                        remaining = remaining - (dot / norm_sq) * b_vec
                if np.linalg.norm(remaining) > tol:
                    kept.append(t)
                    basis.append(remaining.copy())
        
        if len(kept) < 42:
            new_rank = len(kept)
            U_new = [[U[i][t] for t in kept] for i in range(n * m)]
            V_new = [[V[i][t] for t in kept] for i in range(m * p)]
            W_new = [[W[i][t] for t in kept] for i in range(p * n)]
            
            # Verify the decomposition still works
            Uv = np.array(U_new, dtype=float)
            Vv = np.array(V_new, dtype=float)
            Wv = np.array(W_new, dtype=float)
            
            # Reconstruct tensor and compare
            def reconstruct(Um, Vm, Wm, rank):
                T = np.zeros((n*m, m*p, p*n))
                for t in range(rank):
                    T += np.outer(np.outer(Um[:, t], Vm[:, t]), Wm[:, t])
                return T
            
            T_orig = reconstruct(U_arr, V_arr, W_arr, 42)
            T_new = reconstruct(Uv, Vv, Wv, new_rank)
            
            if np.allclose(T_orig, T_new, atol=1e-9):
                return {"U": U_new, "V": V_new, "W": W_new}
    
    # Return the original rank-42 decomposition
    return {"U": U, "V": V, "W": W}
