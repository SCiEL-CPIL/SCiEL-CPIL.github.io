import time
import numpy as np

DIMENSIONS = [2, 5, 6]

_START_TIME = time.time()
TIME_LIMIT = 140.0  # seconds, leaving 10s margin

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook decomposition as baseline (Rank 60)
    # U, V, W are half-integers, so U', V', W' are integers.
    # Equation: sum U' V' W' = 8 * delta
    
    # Schoolbook:
    # For each (i, j, k), we have a column c.
    # U'[i*m+j, c] = 1
    # V'[j*p+k, c] = 1
    # W'[k*n+i, c] = 8
    # Sum: 1 * 1 * 8 = 8. Correct.
    
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
    
    # Try to find lower rank decompositions
    # We search for integer U', V', W' such that sum U' V' W' = 8 delta
    
    def verify(U_p, V_p, W_p, r):
        # Check if sum_c U_p[i*m+j1, c] V_p[j2*p+k, c] W_p[k*n+i, c] == 8 * delta(j1, j2)
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = 0
                        for c in range(r):
                            val += U_p[i*m+j1, c] * V_p[j2*p+k, c] * W_p[k*n+i, c]
                        expected = 8 if j1 == j2 else 0
                        if val != expected:
                            return False
        return True

    def try_rank(r, num_trials=500):
        nonlocal best_U_prime, best_V_prime, best_W_prime, best_rank
        if r >= best_rank:
            return False
        if _time_remaining() < 1.0:
            return False
        
        # Precompute target vector b for each (i, k)
        # b is a vector of length m*m = 25
        # b[j1*m + j2] = 8 if j1==j2 else 0
        b_vec = np.zeros(m * m, dtype=int)
        for j in range(m):
            b_vec[j * m + j] = 8
            
        for trial in range(num_trials):
            if _time_remaining() < 1.0:
                return False
            
            # Generate random integer U', V' with small entries
            # Using {-1, 0, 1} helps with keeping numbers small
            U_t = np.zeros((nm, r), dtype=int)
            V_t = np.zeros((mp, r), dtype=int)
            
            # Fill with random -1, 0, 1
            # Ensure each column has some non-zeros to avoid singular M
            for c in range(r):
                # Pick random rows
                u_rows = np.random.choice(nm, size=min(5, nm), replace=False)
                for row in u_rows:
                    U_t[row, c] = np.random.choice([-1, 1])
                    
                v_rows = np.random.choice(mp, size=min(5, mp), replace=False)
                for row in v_rows:
                    V_t[row, c] = np.random.choice([-1, 1])
            
            # Solve for W'
            W_t = np.zeros((pn, r), dtype=int)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Construct M (25 x r)
                    M = np.zeros((m * m, r), dtype=int)
                    for j1 in range(m):
                        for j2 in range(m):
                            idx = j1 * m + j2
                            for c in range(r):
                                M[idx, c] = U_t[i * m + j1, c] * V_t[j2 * p + k, c]
                    
                    # Solve M w = b_vec for integer w
                    # Use least squares to get real solution, then round
                    try:
                        w_real, residuals, rank_M, sv = np.linalg.lstsq(M, b_vec, rcond=None)
                        w_int = np.round(w_real).astype(int)
                        
                        # Check if integer solution is exact
                        # M @ w_int should equal b_vec
                        if not np.allclose(M @ w_int, b_vec, atol=0.1):
                            valid = False
                            break
                            
                        W_t[k * n + i, :] = w_int
                    except:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # Final verification to be safe
            if verify(U_t, V_t, W_t, r):
                best_U_prime, best_V_prime, best_W_prime = U_t, V_t, W_t
                best_rank = r
                return True
                
        return False

    # Try ranks from lower bound up to schoolbook
    # Lower bound is 30
    for r in range(30, best_rank):
        if _time_remaining() < 2.0:
            break
        # Allocate more trials for lower ranks as they are harder
        trials = 200 if r < 40 else 50
        if try_rank(r, trials):
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
