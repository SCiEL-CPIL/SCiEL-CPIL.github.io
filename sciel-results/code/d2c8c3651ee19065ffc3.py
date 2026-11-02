import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def validate_fast(U, V, W):
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T_recon, 1.0, atol=1e-9):
            return False
        return True

    def try_structured_search(target_rank, max_attempts=5000):
        """
        Try to find a decomposition by fixing the first r-1 columns randomly
        and solving for the r-th column.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random integer matrices for the first r-1 columns
            # and the last column of U and V.
            # We use small integers to increase the chance of consistency.
            
            # U: (n*m, r), V: (m*p, r), W: (p*n, r)
            # We fix U[:, :r-1], V[:, :r-1], W[:, :r-1] randomly.
            # We fix U[:, r-1], V[:, r-1] randomly.
            # We solve for W[:, r-1].
            
            U_prev = np.random.randint(-2, 3, size=(n * m, target_rank - 1))
            V_prev = np.random.randint(-2, 3, size=(m * p, target_rank - 1))
            W_prev = np.random.randint(-2, 3, size=(p * n, target_rank - 1))
            
            U_last = np.random.randint(-2, 3, size=(n * m, 1))
            V_last = np.random.randint(-2, 3, size=(m * p, 1))
            
            # Compute T_prev = sum_{r'=0}^{r-2} U_prev[:, r'] V_prev[:, r'] W_prev[:, r']
            # Reshape for einsum
            U_prev_3d = U_prev.reshape(n, m, target_rank - 1)
            V_prev_3d = V_prev.reshape(m, p, target_rank - 1)
            W_prev_3d = W_prev.reshape(p, n, target_rank - 1)
            
            T_prev = np.einsum('ijr,jkr,kir->ijk', U_prev_3d, V_prev_3d, W_prev_3d)
            
            # RHS = 1 - T_prev
            RHS = 1.0 - T_prev
            
            # Now solve for W_last (size p*n)
            # For each k, i:
            # Sum_j U_last[i*m+j, 0] * V_last[j*p+k, 0] * W_last[k*n+i, 0] = RHS[i,j,k]
            # Let C_ijk = U_last[i*m+j, 0] * V_last[j*p+k, 0]
            # C_ijk * W_last[k*n+i, 0] = RHS[i,j,k]
            
            W_last_candidate = np.zeros((p * n), dtype=float)
            valid = True
            
            for k in range(p):
                for i in range(n):
                    w_val = None
                    for j in range(m):
                        c_val = U_last[i * m + j, 0] * V_last[j * p + k, 0]
                        rhs_val = RHS[i, j, k]
                        
                        if c_val == 0:
                            if abs(rhs_val) > 1e-9:
                                valid = False
                                break
                        else:
                            # w_val = rhs_val / c_val
                            # Check if it's an integer or half-integer
                            # We want exact integer/half-integer.
                            # Let's check if 2 * rhs_val is divisible by c_val?
                            # Or just check if rhs_val / c_val is close to an integer or half-integer.
                            
                            ratio = rhs_val / c_val
                            # Check if ratio is close to an integer
                            nearest_int = round(ratio)
                            if abs(ratio - nearest_int) < 1e-9:
                                candidate = nearest_int
                            else:
                                # Check if ratio is close to a half-integer
                                nearest_half = round(2 * ratio) / 2.0
                                if abs(ratio - nearest_half) < 1e-9:
                                    candidate = nearest_half
                                else:
                                    valid = False
                                    break
                            
                            if w_val is None:
                                w_val = candidate
                            else:
                                if abs(w_val - candidate) > 1e-9:
                                    valid = False
                                    break
                    if not valid:
                        break
                    if w_val is not None:
                        W_last_candidate[k * n + i] = w_val
                if not valid:
                    break
            
            if valid:
                # Construct full U, V, W
                U_full = np.hstack([U_prev, U_last])
                V_full = np.hstack([V_prev, V_last])
                W_full = np.hstack([W_prev, W_last_candidate.reshape(-1, 1)])
                
                # Final validation
                if validate_fast(U_full, V_full, W_full):
                    return {
                        "U": U_full.tolist(),
                        "V": V_full.tolist(),
                        "W": W_full.tolist()
                    }
        
        return None

    # Try ranks from 20 to 29
    for target_rank in range(20, 30):
        if time.time() > deadline:
            break
        
        # Try structured search
        result = try_structured_search(target_rank, max_attempts=2000)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
