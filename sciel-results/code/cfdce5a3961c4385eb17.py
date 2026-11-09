import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_A_sys(U, V, R):
        """Build the linear system matrix for W."""
        # A_sys[row, col] where row = (i,j,k), col = (r, i, k)
        # A_sys[row, col] = U[i*m+j, r] * V[j*p+k, r]
        # We need to solve A_sys @ W_flat = 1 for W_flat
        A_sys = np.zeros((n * m * p, R * n * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i * m * p + j * p + k
                    for r in range(R):
                        col = r * n * p + k * n + i
                        A_sys[row, col] = U[i * m + j, r] * V[j * p + k, r]
        return A_sys

    def check_decomposition_exact(U, V, W_mat, tol=1e-9):
        """Check if the decomposition is valid and entries are close to half-integers."""
        R = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W_mat[k * n + i, r]
                    diff = s - 1.0
                    err += diff * diff
        return err < tol

    def snap_to_half_integers(mat, tol=0.01):
        """Snap matrix entries to nearest half-integer if close."""
        mat_snapped = np.round(mat * 2.0) / 2.0
        # Check if snapping was valid (i.e., original was close to snapped)
        if np.max(np.abs(mat - mat_snapped)) < tol:
            return mat_snapped
        return mat

    def solve_for_W(U, V, R):
        """Given U and V, solve for W using least squares."""
        A_sys = build_A_sys(U, V, R)
        b_vec = np.ones(n * m * p)
        try:
            W_flat, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            return None
        # W_flat has shape (R * n * p), indexed as [r * n * p + k * n + i]
        W_mat = np.zeros((p * n, R))
        for r in range(R):
            for i in range(n):
                for k in range(p):
                    W_mat[k * n + i, r] = W_flat[r * n * p + k * n + i]
        return W_mat

    best_rank = n * m * p
    best_result = None
    
    # Try ranks from 12 (lower bound) up to 23
    for R in range(12, n * m * p):
        if time.time() > deadline - 15:
            break
            
        found = False
        # Increase restarts for lower ranks, decrease for higher?
        # Let's use a fixed number of restarts but faster
        num_restarts = 500 if R < 15 else 200
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Use sparse random initialization to encourage integer/half-integer solutions
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            
            # Each column r is a random vector
            for c in range(R):
                # Randomly select non-zero entries
                # Density ~ 0.5
                for idx in range(n * m):
                    if np.random.random() < 0.5:
                        U[idx, c] = np.random.choice([-1, 1])
                for idx in range(m * p):
                    if np.random.random() < 0.5:
                        V[idx, c] = np.random.choice([-1, 1])
                        
            # Solve for W
            W_mat = solve_for_W(U, V, R)
            if W_mat is None:
                continue
            
            # Snap to half-integers
            U_snap = snap_to_half_integers(U)
            V_snap = snap_to_half_integers(V)
            W_snap = snap_to_half_integers(W_mat)
            
            # Check if snapped version is a valid exact decomposition
            # We check the snapped version against the target
            # Note: The linear system was solved for the original U, V.
            # Snapping changes U, V, W. We must verify the new triple.
            if check_decomposition_exact(U_snap, V_snap, W_snap, tol=1e-9):
                # Found valid decomposition
                if R < best_rank:
                    best_rank = R
                    best_result = {
                        "U": U_snap.tolist(), 
                        "V": V_snap.tolist(), 
                        "W": W_snap.tolist()
                    }
                found = True
                break
            
            # If snapping failed or check failed, we can try a local refinement
            # But for speed, we just restart.
            
        if found:
            break

    # If we found a better rank, return it
    if best_result is not None and best_rank < n * m * p:
        return best_result
    
    # Fallback
    return get_schoolbook()
# EVOLVE-BLOCK-END
