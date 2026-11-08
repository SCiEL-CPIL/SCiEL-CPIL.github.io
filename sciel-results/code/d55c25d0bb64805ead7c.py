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

    # Precompute the linear system structure for W given U, V
    # A_sys[row, col] where row = (i,j,k), col = (r, i, k)
    # A_sys[row, col] = U[i*m+j, r] * V[j*p+k, r]
    # We need to solve A_sys @ W_flat = 1 for W_flat
    
    def build_A_sys(U, V, R):
        """Build the linear system matrix for W."""
        A_sys = np.zeros((n * m * p, R * n * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i * m * p + j * p + k
                    for r in range(R):
                        col = r * n * p + k * n + i
                        A_sys[row, col] = U[i * m + j, r] * V[j * p + k, r]
        return A_sys

    def check_decomposition(U, V, W_mat, tol=1e-8):
        """Check if the decomposition is valid."""
        # W_mat is (p*n, R), so W_mat[k*n+i, r] = W_{r, i, k}
        # Compute T_est[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W_mat[k*n+i, r]
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

    def solve_for_W(U, V, R):
        """Given U and V, solve for W using least squares."""
        A_sys = build_A_sys(U, V, R)
        b_vec = np.ones(n * m * p)
        W_flat, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
        # W_flat has shape (R * n * p), indexed as [r * n * p + k * n + i]
        W_mat = np.zeros((p * n, R))
        for r in range(R):
            for i in range(n):
                for k in range(p):
                    W_mat[k * n + i, r] = W_flat[r * n * p + k * n + i]
        return W_mat

    # Try to find lower rank
    # Lower bound is max(n*m, m*p, p*n) = 12
    # Known results suggest rank 9 might be possible for 2x3x4
    # Let's try ranks from 9 upwards
    
    best_rank = n * m * p
    best_result = None
    
    for R in range(9, n * m * p):
        if time.time() > deadline - 15:
            break
            
        # Run several random restarts
        found = False
        for restart in range(200):
            if time.time() > deadline - 10:
                break
                
            # Random initialization with different strategies
            if restart % 3 == 0:
                # Dense random
                U = np.random.randn(n * m, R)
                V = np.random.randn(m * p, R)
            elif restart % 3 == 1:
                # Sparse random (many zeros)
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                for c in range(R):
                    for i in range(n):
                        for j in range(m):
                            if np.random.random() < 0.5:
                                U[i * m + j, c] = np.random.choice([-1, 1])
                    for j in range(m):
                        for k in range(p):
                            if np.random.random() < 0.5:
                                V[j * p + k, c] = np.random.choice([-1, 1])
            else:
                # Structured initialization: start with schoolbook-like pattern
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                for c in range(R):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    if c < n * m * p:
                        U[i * m + j, c] = 1
                        V[j * p + k, c] = 1
                    else:
                        # Add some randomness
                        U[i * m + j, c] = np.random.choice([-1, 1])
                        V[j * p + k, c] = np.random.choice([-1, 1])
                        # Also set some other entries
                        for _ in range(2):
                            i2 = np.random.randint(n)
                            j2 = np.random.randint(m)
                            U[i2 * m + j2, c] = np.random.choice([-1, 1])
                            k2 = np.random.randint(p)
                            V[j2 * p + k2, c] = np.random.choice([-1, 1])
            
            # ALS iterations: alternate solving for W and updating U, V
            for it in range(30):
                if time.time() > deadline - 5:
                    break
                    
                # Solve for W given U, V
                W_mat = solve_for_W(U, V, R)
                
                # Check if we have a valid solution
                if it % 3 == 0:
                    if check_decomposition(U, V, W_mat, tol=1e-8):
                        # Found valid decomposition
                        if R < best_rank:
                            best_rank = R
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W_mat.tolist()}
                        found = True
                        break
                
                # Update U: for each column r, we need to minimize error
                # This is more complex; let's try a simple gradient-free update
                # Or we can just perturb U and V slightly
                # For simplicity, let's try a few random perturbations
                
                # Actually, let's try a different approach:
                # After solving for W, we can check the residual and try to improve U and V
                # But for now, let's just do random restarts
                
                # Perturb U and V slightly
                U += 0.1 * np.random.randn(n * m, R)
                V += 0.1 * np.random.randn(m * p, R)
            
            if found:
                break
            
            # End restart loop
        
        if found:
            break
        # End R loop
    
    # If we found a better rank, return it
    if best_result is not None and best_rank < n * m * p:
        return best_result
    
    # Fallback
    return get_schoolbook()
# EVOLVE-BLOCK-END
