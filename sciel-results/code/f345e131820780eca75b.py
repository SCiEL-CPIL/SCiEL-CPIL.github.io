"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    schoolbook_rank = dim_u * p  # n*m*p
    
    # Build the tensor as a sparse representation for checking
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We will store the target tensor entries as a list of (i, j, k, val)
    # Or better, build the 3D array if it fits in memory (12*24*18 = 5184 floats)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Function to check if U, V, W decompose T
    def check_decomposition(U, V, W, rank):
        # U: (dim_u, rank), V: (dim_v, rank), W: (dim_w, rank)
        # Reconstruct T
        # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # Check if close to T
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # Only check non-zero entries of T to be faster
        # Find non-zero indices
        nz_u, nz_v, nz_w = np.nonzero(T)
        if len(nz_u) == 0:
            return True
        vals = T_recon[nz_u, nz_v, nz_w]
        # Check if vals are close to 1.0
        # Also check if other entries are close to 0.0
        # For efficiency, we only check the non-zero part strictly
        # and sample the zero part
        if not np.allclose(vals, 1.0, atol=1e-8):
            return False
        # Check a few zero entries
        # Create a random mask of zero entries
        mask_zero = (T == 0)
        if np.any(mask_zero):
            zero_indices = np.argwhere(mask_zero)
            if len(zero_indices) > 0:
                # Sample up to 1000 zero entries
                sample_size = min(1000, len(zero_indices))
                sample_idx = np.random.choice(len(zero_indices), sample_size, replace=False)
                zero_u = zero_indices[sample_idx, 0]
                zero_v = zero_indices[sample_idx, 1]
                zero_w = zero_indices[sample_idx, 2]
                zero_vals = T_recon[zero_u, zero_v, zero_w]
                if not np.allclose(zero_vals, 0.0, atol=1e-8):
                    return False
        return True

    # Start with schoolbook
    def get_schoolbook(rank):
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook(schoolbook_rank)
    
    # Try to reduce rank
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Time budget
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Try ranks from best_rank down to max(1, best_rank - 10)
    # We use a simple iterative refinement if we start from a valid rank
    # But reducing rank is hard.
    # Instead, we can try to find a rank < schoolbook_rank using random initialization and ALS
    # Given the small size, we can try a few random restarts for rank 71, 70
    
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 15, 0), -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts for this rank
        success = False
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize
            U = np.random.randn(dim_u, target_rank)
            V = np.random.randn(dim_v, target_rank)
            W = np.random.randn(dim_w, target_rank)
            
            # Scale to match norms
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                # U * (V^T W) = T_unfolded
                # T_unfolded (dim_u, dim_v*dim_w)
                # We can solve this by least squares
                # T_mat = T.reshape(dim_u, dim_v * dim_w)
                # M = V.T @ W -> (dim_v*dim_w, target_rank)? No.
                # We want U such that U (V^T W) approx T_mat
                # This is a linear system: U * M = T_mat
                # Where M is (target_rank, dim_v*dim_w)
                # V.T is (target_rank, dim_v)
                # W is (dim_w, target_rank)
                # We need the element-wise product? No, tensor product.
                # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
                # Fix V, W. Update U.
                # For each i, j, k: T_{ijk} = sum_r U_{ir} (V_{jr} W_{kr})
                # This is a linear system for U_{ir}.
                # Let M_{jk, r} = V_{jr} W_{kr}. Size (dim_v*dim_w, target_rank).
                # Then U_{i, :} M_{:, :} = T_{i, :} (flattened over j,k)
                
                M = (V.T * W).reshape(dim_v * dim_w, target_rank)
                T_mat = T.reshape(dim_u, dim_v * dim_w).T # (dim_v*dim_w, dim_u)
                # We want U (dim_u, target_rank) such that U @ M.T = T_mat.T ?
                # U_{i, r} M_{j k, r} = T_{i, j k}
                # Sum_r U_{i, r} M_{j k, r} = T_{i, j k}
                # This is (U @ M.T)_{i, j k} = T_{i, j k}
                # So U @ M.T = T.reshape(dim_u, dim_v * dim_w)
                # U = T.reshape(dim_u, dim_v * dim_w) @ pinv(M.T)
                
                T_u = T.reshape(dim_u, dim_v * dim_w)
                try:
                    U_new = T_u @ np.linalg.pinv(M.T)
                except:
                    break
                
                # Update V
                M_v = (U.T * W).reshape(dim_u * dim_w, target_rank)
                T_v = T.reshape(dim_v, dim_u * dim_w) # Need correct unfolding
                # T_{j, i k} = sum_r V_{j, r} U_{i, r} W_{k, r}
                # V @ M_v.T = T_v
                T_v_mat = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                try:
                    V_new = T_v_mat @ np.linalg.pinv(M_v.T)
                except:
                    break
                
                # Update W
                M_w = (U.T * V).reshape(dim_u * dim_v, target_rank)
                T_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                try:
                    W_new = T_w @ np.linalg.pinv(M_w.T)
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
            # Check if this solution is valid
            if check_decomposition(U, V, W, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                success = True
                break # Found valid rank, try lower? Or stop?
                # We want the minimal, so continue to lower ranks if time permits
        
    # If no better found, return schoolbook
    if best_rank == schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Ensure integer/half-integer if possible? The problem says "integer/half-integer entries"
        # My random solution might have floats.
        # The task says "entries that exactly decompose...".
        # If the solution is exact, it should be close to integers.
        # We can round them.
        best_U = np.round(best_U)
        best_V = np.round(best_V)
        best_W = np.round(best_W)
        if check_decomposition(best_U, best_V, best_W, best_rank):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        else:
            # Fallback to schoolbook
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
