"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W

    def validate(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly represents matrix multiplication."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Try to find a lower rank using randomized search for integer/half-integer solutions
    # Strategy: Fix random integer U and V, solve linear system for W, check if W is half-integer.
    
    def try_rank(R):
        if time.time() > deadline - 20:
            return None
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Generate random integer U, V with small entries
            U_rand = np.random.randint(-1, 2, (n * m, R))
            V_rand = np.random.randint(-1, 2, (m * p, R))
            
            # Build linear system A x = b for W
            # x is W flattened (p*n * R)
            num_eqs = n * m * p
            num_vars = p * n * R
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        for r in range(R):
                            A[idx, w_idx * R + r] = U_rand[u_idx, r] * V_rand[v_idx, r]
                        idx += 1
            
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                W_rand = W_flat.reshape(p * n, R)
                
                # Check if W_rand is close to half-integers
                W_check = W_rand * 2
                # Check distance to nearest integer
                dist = np.abs(W_check - np.round(W_check))
                if np.max(dist) < 1e-5:
                    W_int = np.round(W_check) / 2.0
                    U_int = U_rand.astype(float)
                    V_int = V_rand.astype(float)
                    
                    # Convert to lists for validation
                    if validate(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            except Exception:
                pass
                
        return None

    # Try ranks 74 down to 70
    for R in range(rank_schoolbook - 1, 69, -1):
        if time.time() > deadline - 30:
            break
            
        result = try_rank(R)
        if result:
            return result
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
