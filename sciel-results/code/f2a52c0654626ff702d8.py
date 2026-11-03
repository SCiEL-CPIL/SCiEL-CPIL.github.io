"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    # Initialize Schoolbook
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1

    # Target Tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    # Try to find a lower rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school

    start_time = time.time()
    deadline = 140.0

    # Search ranks from best_rank - 1 down to some floor
    # We'll use a simple randomized iterative method (like ALS or Gradient Descent on flattened vectors)
    # Given the time limit, we can only do a few tries.
    
    for target_rank in range(best_rank - 1, max(best_rank - 5, 10), -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Initialize with schoolbook terms, then randomly perturb/merge
            # A better init: Take first target_rank terms of schoolbook, then add random noise?
            # Or: Initialize U, V, W with random values scaled to match T's norm.
            
            # Let's try: Start with schoolbook, pick 'target_rank' columns, and then optimize.
            # Actually, just optimizing fixed rank random init is better for escaping local minima.
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Simple Gradient Descent / ALS
            # Minimize || T - U V W ||_F^2
            # Unroll T to (n*m, m*p, p*n)
            # For a fixed U, V, the optimal W is (U^T T V)^T / (U^T U * V^T V) element-wise? No, that's not right for tensor.
            # Tensor ALS:
            # W_{k,n+i, c} = sum_{i,j} T_{i*m+j, j*p+k, k*n+i} U_{i*m+j, c} V_{j*p+k, c} / sum_{i,j} (U_{i*m+j, c} V_{j*p+k, c})^2
            
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Update U
                # U_{i*m+j, c} = sum_{k} T_{i*m+j, j*p+k, k*n+i} V_{j*p+k, c} W_{k*n+i, c} / sum_{k} (V_{j*p+k, c} W_{k*n+i, c})^2
                for c in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            num = 0
                            den = 0
                            for k in range(p):
                                num += T[i * m + j, j * p + k, k * n + i] * V[j * p + k, c] * W[k * n + i, c]
                                den += V[j * p + k, c] * W[k * n + i, c]
                            if den != 0:
                                U[i * m + j, c] = num / den
                            else:
                                U[i * m + j, c] = 0.1 # avoid zero division
                    
                # Update V
                for c in range(target_rank):
                    for j in range(m):
                        for k in range(p):
                            num = 0
                            den = 0
                            for i in range(n):
                                num += T[i * m + j, j * p + k, k * n + i] * U[i * m + j, c] * W[k * n + i, c]
                                den += U[i * m + j, c] * W[k * n + i, c]
                            if den != 0:
                                V[j * p + k, c] = num / den
                            else:
                                V[j * p + k, c] = 0.1
                                
                # Update W
                for c in range(target_rank):
                    for k in range(p):
                        for i in range(n):
                            num = 0
                            den = 0
                            for j in range(m):
                                num += T[i * m + j, j * p + k, k * n + i] * U[i * m + j, c] * V[j * p + k, c]
                                den += U[i * m + j, c] * V[j * p + k, c]
                            if den != 0:
                                W[k * n + i, c] = num / den
                            else:
                                W[k * n + i, c] = 0.1
                                
            # Check error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        error += (val - T[i * m + j, j * p + k, k * n + i]) ** 2
            
            if error < 1e-6:
                # Found a valid decomposition of rank target_rank
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break # Move to next lower rank
        
        # If we found a better rank, continue trying to go lower
        # If not, break and return best found

    # Convert to lists of floats (or ints if close)
    # The task asks for integer/half-integer entries.
    # Random search will produce floats.
    # We can try to round them to nearest 0.5 and check validity.
    def clean_and_check(U, V, W, T, n, m, p):
        U_clean = np.round(U * 2) / 2
        V_clean = np.round(V * 2) / 2
        W_clean = np.round(W * 2) / 2
        
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U.shape[1]):
                        val += U_clean[i * m + j, c] * V_clean[j * p + k, c] * W_clean[k * n + i, c]
                    error += (val - T[i * m + j, j * p + k, k * n + i]) ** 2
        return error < 1e-4, U_clean, V_clean, W_clean

    # If we found a low rank, try to clean it
    if best_rank < rank_school:
        err, Uc, Vc, Wc = clean_and_check(best_U, best_V, best_W, T, n, m, p)
        if err < 1e-4:
            best_U, best_V, best_W = Uc, Vc, Wc
        else:
            # If cleaning fails, revert to schoolbook to ensure validity with simple numbers
            # Or keep the float values? "integer/half-integer entries" is a constraint.
            # If I can't satisfy it, I should return a valid schoolbook.
            best_U, best_V, best_W = U_sb, V_sb, W_sb
            best_rank = rank_school

    # Convert numpy to list of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    # Ensure they are exactly 0 or 1 for schoolbook, or close for others
    # For schoolbook, they are already 0/1.
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
