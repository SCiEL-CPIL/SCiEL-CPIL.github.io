import time
import numpy as np

DIMENSIONS = [2, 5, 6]

_START_TIME = time.time()
TIME_LIMIT = 145.0  # seconds, leaving 5s margin

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition as baseline (Rank 60)
    rank_school = n * m * p
    U_prime = np.zeros((nm, rank_school), dtype=int)
    V_prime = np.zeros((mp, rank_school), dtype=int)
    W_prime = np.zeros((pn, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_prime[i * m + j, col] = 1
                V_prime[j * p + k, col] = 1
                W_prime[k * n + i, col] = 8
                col += 1
                
    best_U_prime, best_V_prime, best_W_prime = U_prime, V_prime, W_prime
    best_rank = rank_school
    
    # Precompute target vector b for each (i, k) is not needed for verify, 
    # but for solving W we need it.
    b_vec = np.zeros(m * m, dtype=int)
    for j in range(m):
        b_vec[j * m + j] = 8

    def verify_fast(U_p, V_p, W_p, r):
        # Reshape to 3D
        # U_p: (nm, r) -> (n, m, r)
        # V_p: (mp, r) -> (m, p, r)
        # W_p: (pn, r) -> (p, n, r)
        U_3d = U_p.reshape(n, m, r)
        V_3d = V_p.reshape(m, p, r)
        W_3d = W_p.reshape(p, n, r)
        
        # Check all i, k
        for i in range(n):
            for k in range(p):
                # A[j1, c] = U[i, j1, c] * W[k, i, c]
                # W_3d[k, i, :] is a vector of length r
                w_vec = W_3d[k, i, :]
                a_mat = U_3d[i, :, :] * w_vec # (m, r)
                
                # V_3d[:, k, :] is (m, r)
                v_mat = V_3d[:, k, :] # (m, r)
                
                # M[j1, j2] = sum_c a_mat[j1, c] * v_mat[j2, c]
                # M = a_mat @ v_mat.T
                M = a_mat @ v_mat.T
                
                # Check if M == 8 * I_m
                if not np.all(M == 8 * np.eye(m, dtype=int)):
                    return False
        return True

    def try_rank(r, num_trials=100):
        nonlocal best_U_prime, best_V_prime, best_W_prime, best_rank
        if r >= best_rank:
            return False
        if _time_remaining() < 2.0:
            return False
        
        for trial in range(num_trials):
            if _time_remaining() < 1.0:
                return False
            
            # Generate random integer U', V'
            # Use small integers to keep W small
            U_t = np.random.randint(-2, 3, size=(nm, r), dtype=int)
            V_t = np.random.randint(-2, 3, size=(mp, r), dtype=int)
            
            # Solve for W'
            W_t = np.zeros((pn, r), dtype=int)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Construct M (m*m x r)
                    # M[j1*m + j2, c] = U[i, j1, c] * V[j2, k, c]
                    M = np.zeros((m * m, r), dtype=int)
                    for j1 in range(m):
                        for j2 in range(m):
                            idx = j1 * m + j2
                            M[idx, :] = U_t[i * m + j1, :] * V_t[j2 * p + k, :]
                    
                    # Solve M w = b_vec
                    # Use least squares
                    try:
                        w_real, residuals, rank_M, sv = np.linalg.lstsq(M, b_vec, rcond=None)
                        w_int = np.round(w_real).astype(int)
                        
                        # Check exactness
                        if not np.all(M @ w_int == b_vec):
                            valid = False
                            break
                            
                        W_t[k * n + i, :] = w_int
                    except:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # Final verification
            if verify_fast(U_t, V_t, W_t, r):
                best_U_prime, best_V_prime, best_W_prime = U_t, V_t, W_t
                best_rank = r
                return True
                
        return False

    # Try ranks from lower bound up to schoolbook
    # Lower bound for 2x5x6 is at least 30 (roughly nm*p/2)
    # We start from 40 to be safe and fast, then go down if time permits?
    # No, we want the lowest rank. We should try from low to high.
    # But low ranks are harder.
    # Strategy: Try 59, 58, ... down to 40.
    # If we find one, we can try to go lower.
    
    # Given the difficulty, we might not find anything below 60.
    # Let's try a few ranks near 60 to see if we can improve.
    for r in range(59, 39, -1):
        if _time_remaining() < 5.0:
            break
        # More trials for lower ranks? No, lower ranks are harder, so maybe fewer trials but smarter?
        # For now, fixed trials.
        if try_rank(r, num_trials=50):
            break

    # Convert to half-integers (floats) for output
    U_out = best_U_prime / 2.0
    V_out = best_V_prime / 2.0
    W_out = best_W_prime / 2.0

    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
