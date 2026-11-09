import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # A: n x m, B: m x p, C: n x p
    
    def verify_decomp(U, V, W, rank):
        """
        Verifies if the given U, V, W matrices exactly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Condition: For all i, j, k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Pre-compute indices for speed
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Vectorized dot product
                    s = np.dot(U[u_idx, :], np.dot(V[v_idx, :], W[w_idx, :]))
                    # Actually, it's sum_r U[u,r] * V[v,r] * W[w,r]
                    # np.dot(U[u,:], V[v,:]*W[w,:]) is faster
                    s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def round_to_half(arr):
        return np.round(arr * 2) / 2.0

    def get_rank_17_solution():
        """
        Returns a known valid rank-17 decomposition.
        """
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Term 1
        U[0, 0] += 1; U[4, 0] += 1
        V[0, 0] += 1; V[4, 0] += 1
        W[0, 0] += 1; W[3, 0] += 1
        
        # Term 2
        U[3, 1] += 1; U[4, 1] += 1
        V[0, 1] += 1
        W[1, 1] += 1; W[3, 1] -= 1
        
        # Term 3
        U[0, 2] += 1
        V[1, 2] += 1; V[4, 2] -= 1
        W[2, 2] += 1; W[3, 2] += 1
        
        # Term 4
        U[4, 3] += 1
        V[3, 3] += 1; V[0, 3] -= 1
        W[0, 3] += 1; W[1, 3] += 1
        
        # Term 5
        U[0, 4] += 1; U[1, 4] += 1
        V[4, 4] += 1
        W[0, 4] -= 1; W[2, 4] += 1
        
        # Term 6
        U[3, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        W[3, 5] += 1
        
        # Term 7
        U[1, 6] += 1; U[4, 6] -= 1
        V[3, 6] += 1; V[4, 6] += 1
        W[0, 6] += 1
        
        # Term 8: A00B02
        U[0, 7] += 1
        V[2, 7] += 1
        W[4, 7] += 1
        
        # Term 9: A01B12
        U[1, 8] += 1
        V[5, 8] += 1
        W[4, 8] += 1
        
        # Term 10: A10B02
        U[3, 9] += 1
        V[2, 9] += 1
        W[5, 9] += 1
        
        # Term 11: A11B12
        U[4, 10] += 1
        V[5, 10] += 1
        W[5, 10] += 1
        
        # Term 12: A02B20
        U[2, 11] += 1
        V[6, 11] += 1
        W[0, 11] += 1
        
        # Term 13: A02B21
        U[2, 12] += 1
        V[7, 12] += 1
        W[2, 12] += 1
        
        # Term 14: A02B22
        U[2, 13] += 1
        V[8, 13] += 1
        W[4, 13] += 1
        
        # Term 15: A12B20
        U[5, 14] += 1
        V[6, 14] += 1
        W[1, 14] += 1
        
        # Term 16: A12B21
        U[5, 15] += 1
        V[7, 15] += 1
        W[3, 15] += 1
        
        # Term 17: A12B22
        U[5, 16] += 1
        V[8, 16] += 1
        W[5, 16] += 1
        
        return U, V, W

    def try_rank_16():
        """
        Attempts to find a rank-16 decomposition using multi-start optimization.
        """
        if time.time() > deadline - 10:
            return None
            
        rank = 16
        # Number of variables: n*m*rank + m*p*rank + p*n*rank
        num_vars = (n * m + m * p + p * n) * rank
        
        def error_func(params):
            U_flat = params[0:n*m*rank]
            V_flat = params[n*m*rank:n*m*rank+m*p*rank]
            W_flat = params[n*m*rank+m*p*rank:]
            
            U_ = U_flat.reshape(n*m, rank)
            V_ = V_flat.reshape(m*p, rank)
            W_ = W_flat.reshape(p*n, rank)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = np.dot(U_[u_idx, :], V_[v_idx, :] * W_[w_idx, :])
                        err += (s - 1.0)**2
            return err

        # Multiple random restarts
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Scale: entries are typically small integers or halves.
            # Start with small random values to avoid huge gradients.
            np.random.seed(42 + attempt)
            U0 = np.random.randn(n * m, rank) * 0.5
            V0 = np.random.randn(m * p, rank) * 0.5
            W0 = np.random.randn(p * n, rank) * 0.5
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # Use L-BFGS-B for gradient-based optimization
                # It is generally faster than Powell for smooth functions.
                res = minimize(error_func, params0, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    U_sol = res.x[0:n*m*rank].reshape(n*m, rank)
                    V_sol = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_sol = res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    # Try to round to half-integers
                    U_rounded = round_to_half(U_sol)
                    V_rounded = round_to_half(V_sol)
                    W_rounded = round_to_half(W_sol)
                    
                    if verify_decomp(U_rounded, V_rounded, W_rounded, rank):
                        return [U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()]
                    
                    # If rounding fails, maybe the solution is not exactly half-integer?
                    # The problem statement says "integer/half-integer entries".
                    # If the numerical solution is close to half-integers, rounding should work.
                    # If not, we might need a different rounding strategy or accept float?
                    # "exactly decompose" implies the values must be exact.
                    # If the float solution is valid within 1e-6, but not exactly half-integer,
                    # it might not be the intended "exact" algebraic solution.
                    # However, if the problem allows any real numbers that exactly decompose,
                    # we should return the float values.
                    # "integer/half-integer entries" is a constraint on the output format?
                    # "factor matrices ... with integer/half-integer entries".
                    # This implies we MUST return integer or half-integer values.
                    # So if rounding fails, this solution is invalid.
                    
            except Exception:
                continue
                
        return None

    # Try to find a better (lower rank) solution
    result_16 = try_rank_16()
    
    if result_16:
        U, V, W = result_16
        return {"U": U, "V": V, "W": W}
        
    # Fall back to the known rank 17 solution
    U17, V17, W17 = get_rank_17_solution()
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
